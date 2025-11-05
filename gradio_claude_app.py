import gradio as gr
import anthropic
import json
import os
from typing import List, Dict, Any
import asyncio
import threading
from mnist_logistic_regression import train_mnist_model, get_model_info

class ClaudeToolChatbot:
    def __init__(self):
        self.client = None
        self.conversation_history = []
        self.tools = [
            {
                "name": "mnist_logistic_regression",
                "description": "Build and train a logistic regression model on the MNIST dataset using PyTorch",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "epochs": {
                            "type": "integer",
                            "description": "Number of training epochs (default: 10)",
                            "default": 10
                        },
                        "learning_rate": {
                            "type": "number",
                            "description": "Learning rate for training (default: 0.01)",
                            "default": 0.01
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Batch size for training (default: 64)",
                            "default": 64
                        }
                    }
                }
            },
            {
                "name": "get_mnist_model_info",
                "description": "Get information about the MNIST logistic regression model architecture and dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def set_api_key(self, api_key: str) -> str:
        """Set the Anthropic API key"""
        if not api_key or not api_key.strip():
            return "❌ Please enter a valid API key"
        
        try:
            self.client = anthropic.Anthropic(api_key=api_key.strip())
            # Test the API key with a simple request
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return "✅ API key set successfully!"
        except Exception as e:
            self.client = None
            return f"❌ Invalid API key: {str(e)}"
    
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return the result"""
        try:
            if tool_name == "mnist_logistic_regression":
                epochs = tool_input.get("epochs", 10)
                learning_rate = tool_input.get("learning_rate", 0.01)
                batch_size = tool_input.get("batch_size", 64)
                
                result = train_mnist_model(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
                return json.dumps(result, indent=2)
            
            elif tool_name == "get_mnist_model_info":
                result = get_model_info()
                return json.dumps(result, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}"
        
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def chat(self, message: str, history: List[List[str]]) -> tuple:
        """Main chat function"""
        if not self.client:
            return history + [[message, "❌ Please set your API key first using the sidebar."]], ""
        
        if not message.strip():
            return history, ""
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            # Create the Claude message with tools
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                tools=self.tools,
                messages=self.conversation_history
            )
            
            assistant_response = ""
            tool_results = []
            
            # Process the response
            for content_block in response.content:
                if content_block.type == "text":
                    assistant_response += content_block.text
                elif content_block.type == "tool_use":
                    # Execute the tool
                    tool_name = content_block.name
                    tool_input = content_block.input
                    tool_id = content_block.id
                    
                    assistant_response += f"\n🔧 **Using tool: {tool_name}**\n"
                    assistant_response += f"Parameters: {json.dumps(tool_input, indent=2)}\n"
                    
                    # Execute the tool
                    tool_result = self.execute_tool(tool_name, tool_input)
                    tool_results.append({
                        "tool_use_id": tool_id,
                        "content": tool_result
                    })
                    
                    assistant_response += f"\n📊 **Tool Result:**\n```\n{tool_result}\n```\n"
            
            # If there were tool uses, we need to continue the conversation
            if tool_results:
                # Add the assistant's response with tool use
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response.content
                })
                
                # Add tool results
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
                
                # Get Claude's final response
                final_response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    tools=self.tools,
                    messages=self.conversation_history
                )
                
                for content_block in final_response.content:
                    if content_block.type == "text":
                        assistant_response += f"\n\n💬 **Claude's Analysis:**\n{content_block.text}"
                
                # Update conversation history with final response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response.content
                })
            else:
                # Simple text response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
            
            # Update the chat history for display
            updated_history = history + [[message, assistant_response]]
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            updated_history = history + [[message, error_msg]]
        
        return updated_history, ""

def create_app():
    """Create the Gradio application"""
    chatbot_instance = ClaudeToolChatbot()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .sidebar {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-chat {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
    }
    """
    
    with gr.Blocks(css=css, title="Claude Chatbot with Tools") as app:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🤖 Claude Chatbot with MNIST Tools</h1>
            <p>Chat with Claude and use machine learning tools! Try asking: "Can you train a logistic regression model on MNIST?"</p>
        </div>
        """)
        
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.HTML("<h3>⚙️ Settings</h3>")
                
                api_key_input = gr.Textbox(
                    label="Anthropic API Key",
                    type="password",
                    placeholder="Enter your Anthropic API key...",
                    lines=1
                )
                
                set_key_btn = gr.Button("Set API Key", variant="primary")
                api_status = gr.Textbox(
                    label="Status",
                    value="❌ No API key set",
                    interactive=False,
                    lines=2
                )
                
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #e8f4f8; border-radius: 8px;">
                    <h4>🛠️ Available Tools:</h4>
                    <ul>
                        <li><b>MNIST Logistic Regression:</b> Train a PyTorch model</li>
                        <li><b>Model Info:</b> Get dataset and architecture details</li>
                    </ul>
                    <p><i>Just ask Claude to use these tools naturally in conversation!</i></p>
                </div>
                """)
            
            # Main chat area
            with gr.Column(scale=3, elem_classes=["main-chat"]):
                chatbot = gr.Chatbot(
                    height=600,
                    label="Chat with Claude",
                    elem_id="chatbot"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here... (e.g., 'Train a logistic regression model on MNIST with 5 epochs')",
                        lines=2,
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Event handlers
        def handle_api_key(api_key):
            return chatbot_instance.set_api_key(api_key)
        
        def handle_message(message, history):
            return chatbot_instance.chat(message, history)
        
        def clear_chat():
            chatbot_instance.conversation_history = []
            return []
        
        # Connect events
        set_key_btn.click(handle_api_key, inputs=[api_key_input], outputs=[api_status])
        
        send_btn.click(
            handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(clear_chat, outputs=[chatbot])
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
