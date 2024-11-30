#!/bin/bash

# Colors and formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
GRAY="\033[0;90m"
NC="\033[0m"
DIM="\033[2m"

# Default values
MAX_TOKENS=2000
TEMPERATURE=0.7
MODEL_PATH="adapters"
PORT=8080
CONVERSATION_FILE="/tmp/mlx_chat_history.json"

# Ensure clean exit
trap cleanup EXIT

print_welcome() {
    clear
    echo -e "${BOLD}${BLUE}Chat with Your Fine-tuned Model${NC}"
    echo
    if [ -f "adapters/training_info.json" ]; then
        echo -e "${CYAN}Model Details:${NC}"
        echo -e "$(python3 -c 'import json; info=json.load(open("adapters/training_info.json")); print(f"• Base Model: {info["model"]}\n• Training Completed: {info["completed"]}\n• Final Loss: {info["final_loss"]}")')"
        echo
    fi
    
    echo -e "${YELLOW}Available Commands:${NC}"
    echo -e "• ${BOLD}/system${NC}  - Set system prompt for better control"
    echo -e "• ${BOLD}/temp${NC}    - Adjust temperature (default: $TEMPERATURE)"
    echo -e "• ${BOLD}/save${NC}    - Save this conversation"
    echo -e "• ${BOLD}/load${NC}    - Load a previous conversation"
    echo -e "• ${BOLD}/clear${NC}   - Start fresh"
    echo -e "• ${BOLD}/help${NC}    - Show all commands"
    echo -e "• ${BOLD}/quit${NC}    - Exit chat"
    echo
    echo -e "${DIM}Type your message and press Enter. For multi-line input, keep typing and use an empty line to send.${NC}"
    echo
}

init_conversation() {
    echo '{"messages":[]}' > "$CONVERSATION_FILE"
}

add_to_history() {
    local role="$1"
    local content="$2"
    python3 -c "
import json, sys
with open('$CONVERSATION_FILE', 'r+') as f:
    hist = json.load(f)
    hist['messages'].append({'role': '$role', 'content': '''$content'''})
    f.seek(0)
    json.dump(hist, f)
    f.truncate()
"
}

get_conversation() {
    python3 -c "
import json
with open('$CONVERSATION_FILE') as f:
    hist = json.load(f)
    for msg in hist['messages']:
        if msg['role'] == 'user':
            print(f'\n${BOLD}You:${NC} {msg[\"content\"]}')
        elif msg['role'] == 'assistant':
            print(f'\n${BOLD}Assistant:${NC} {msg[\"content\"]}')
        elif msg['role'] == 'system':
            print(f'\n${GRAY}System: {msg[\"content\"]}${NC}')
"
}

start_server() {
    echo -e "${BOLD}Starting model server...${NC}"
    mlx_lm.server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --trust-remote-code \
        --log-level INFO \
        --cache-limit-gb 4 \
        --use-default-chat-template &
    
    SERVER_PID=$!
    
    # Wait for server to start
    for i in {1..10}; do
        if curl -s "localhost:$PORT/v1/models" > /dev/null; then
            echo -e "${GREEN}Server ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    echo -e "${YELLOW}Warning: Server may not be fully ready${NC}"
    return 1
}

stream_response() {
    local response=""
    local chunk_count=0
    
    # Stream response chunks
    while IFS= read -r chunk; do
        if [[ $chunk == data:* ]]; then
            chunk="${chunk#data: }"
            if [ "$chunk" != "[DONE]" ]; then
                # Extract and print the content
                content=$(echo "$chunk" | python3 -c "
import json, sys
try:
    chunk = json.loads(sys.stdin.read())
    if 'choices' in chunk and chunk['choices']:
        content = chunk['choices'][0].get('delta', {}).get('content', '')
        if content:
            print(content, end='', flush=True)
except: pass
")
                response+="$content"
                chunk_count=$((chunk_count + 1))
            fi
        fi
    done
    
    echo
    # Add complete response to history
    add_to_history "assistant" "$response"
}

handle_command() {
    local cmd="$1"
    case "$cmd" in
        "/clear")
            clear
            init_conversation
            print_welcome
            ;;
        "/save")
            local filename="chat_$(date +%Y%m%d_%H%M%S).json"
            cp "$CONVERSATION_FILE" "$filename"
            echo -e "${GREEN}Conversation saved to: $filename${NC}"
            ;;
        "/load")
            echo -e "${YELLOW}Enter path to conversation file:${NC}"
            read -r filename
            if [ -f "$filename" ]; then
                cp "$filename" "$CONVERSATION_FILE"
                clear
                get_conversation
            else
                echo -e "${YELLOW}File not found${NC}"
            fi
            ;;
        "/system")
            echo -e "${YELLOW}Enter system prompt:${NC}"
            read -r system_prompt
            add_to_history "system" "$system_prompt"
            echo -e "${GREEN}System prompt updated${NC}"
            ;;
        "/temp")
            echo -e "${YELLOW}Enter temperature (0.0-1.0):${NC}"
            read -r new_temp
            if [[ "$new_temp" =~ ^[0-9]*\.?[0-9]+$ ]] && (( $(echo "$new_temp <= 1.0" | bc -l) )); then
                TEMPERATURE=$new_temp
                echo -e "${GREEN}Temperature updated to $TEMPERATURE${NC}"
            else
                echo -e "${YELLOW}Invalid temperature value${NC}"
            fi
            ;;
        "/help")
            echo -e "${YELLOW}Commands:${NC}"
            echo -e "• /clear    - Clear conversation history"
            echo -e "• /save     - Save conversation to file"
            echo -e "• /load     - Load previous conversation"
            echo -e "• /system   - Set system prompt"
            echo -e "• /temp     - Adjust temperature (0.0-1.0)"
            echo -e "• /help     - Show this help message"
            echo -e "• /quit     - Exit chat"
            ;;
        "/quit")
            return 1
            ;;
        *)
            return 0
            ;;
    esac
    return 0
}

chat() {
    init_conversation
    local continue_chat=true
    
    # Add default system prompt based on training info
    if [ -f "adapters/training_info.json" ]; then
        local model_name=$(python3 -c 'import json; print(json.load(open("adapters/training_info.json"))["model"])')
        add_to_history "system" "You are a helpful AI assistant based on $model_name, fine-tuned for specific tasks. Be concise and accurate in your responses."
    fi
    
    while $continue_chat; do
        echo -e "${BOLD}You:${NC} ${DIM}(Type a command or your message)${NC}"
        # Read user input, allowing for multi-line input
        message=""
        while IFS= read -r line; do
            [[ -z "$line" ]] && break
            message+="$line"$'\n'
        done
        message=${message%$'\n'} # Remove trailing newline
        
        # Handle empty input
        [[ -z "$message" ]] && continue
        
        # Process commands
        if [[ "$message" == /* ]]; then
            if ! handle_command "$message"; then
                echo -e "${GREEN}Thanks for using your fine-tuned model! To try different settings or models, return to the tutorial.${NC}"
                break
            fi
            continue
        fi
        
        # Add user message to history
        add_to_history "user" "$message"
        
        # Show thinking indicator
        echo -ne "${DIM}Thinking...${NC}"
        
        # Get full conversation history
        local messages=$(python3 -c "
import json
with open('$CONVERSATION_FILE') as f:
    print(json.dumps({'messages': json.load(f)['messages']}))
")
        
        # Clear thinking indicator
        echo -ne "\r\033[K"
        
        # Send request to server with streaming
        echo -ne "${BOLD}Assistant:${NC} "
        curl -sN "localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                $messages,
                \"stream\": true,
                \"temperature\": $TEMPERATURE
            }" | stream_response
        
        echo
    done
}

cleanup() {
    # Kill the server process
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
    fi
    # Clean up temporary files
    rm -f "$CONVERSATION_FILE"
    echo -e "${GREEN}Chat session ended. Server stopped.${NC}"
}

# Main execution
print_welcome
start_server && chat
