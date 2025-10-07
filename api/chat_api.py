import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import argparse
import json
import re
import threading
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from examples.general_rag_demo import initialize_and_process_query
from agno.agent import Agent
from agno.models.anthropic import Claude
import pdb  # For debugging with pdb.set_trace()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Global variables
model_name = "gemini/gemini-2.5-flash-preview-05-20"  # Default value
suggestion_agent = None
suggestion_callbacks = {}  # Store callback URLs for sending suggestions

def get_suggestion_agent():
    """Initialize and return the suggestion agent"""
    global suggestion_agent
    if suggestion_agent is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        suggestion_agent = Agent(
            # model=Claude(id="claude-3-5-sonnet-20240620", api_key=api_key),
            model=Claude(id="claude-opus-4-20250514", api_key=api_key),
            markdown=True,
            add_history_to_messages=True,
            num_history_runs=3,
        )
    return suggestion_agent

def extract_json_from_response(response_text):
    """Extract JSON from agent response text"""
    # First try to find JSON array or object
    json_match = re.search(r'(\[.*?\]|\{.*?\})', response_text, re.DOTALL)
    
    if not json_match:
        raise ValueError("No JSON object or array found in the response.")
    
    json_str = json_match.group(1)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

def generate_suggestions_with_agent(question, response_data):
    """Generate suggestions using the agno Agent with Claude"""
    try:
        agent = get_suggestion_agent()
        
        # Create a comprehensive prompt for suggestion generation
        prompt = f"""
                    Based on the user's question: "{question}"

                    And the system's response: {json.dumps(response_data, indent=2)}

                    Generate 5 helpful and relevant suggestions for follow-up questions or actions the user might want to take next. 

                    Each suggestion should be:
                    - Relevant to the current topic and response
                    - Actionable and specific
                    - Helpful for the user's understanding or next steps

                    Format your response as a JSON array where each suggestion has exactly these fields:
                    - title: A short descriptive title (max 50 characters)
                    - message: pass empty string as default
                    - partial: false (always set to false)
                    - className: "suggestion-item" (always this exact value)

                    Example format:
                    [
                        {{
                            "title": "Document Requirements",
                            "message": "",
                            "partial": false,
                            "className": "suggestion-item"
                        }},
                        {{
                            "title": "Visiting Hours",
                            "message": "",
                            "partial": false,
                            "className": "suggestion-item"
                        }}
                    ]

                    Return only the JSON array, no additional text or explanation.
                    """
        
        # Run the agent to generate suggestions
        agent_response = agent.run(prompt)
        
        # Extract the message content from the agent response
        if hasattr(agent_response, 'content') and agent_response.content:
            response_text = agent_response.content
        elif hasattr(agent_response, 'message') and agent_response.message:
            response_text = agent_response.message
        else:
            response_text = str(agent_response)
        
        # Extract JSON from the response
        suggestions = extract_json_from_response(response_text)
        
        # Validate and clean the suggestions
        if not isinstance(suggestions, list):
            suggestions = []
        
        validated_suggestions = []
        for suggestion in suggestions[:5]:  # Limit to 5 suggestions
            if isinstance(suggestion, dict):
                validated_suggestion = {
                    "title": str(suggestion.get("title", "Suggestion"))[:50],
                    "message": str(suggestion.get("message", ""))[:100],
                    "partial": False,
                    "className": "suggestion-item"
                }
                validated_suggestions.append(validated_suggestion)
        
        return validated_suggestions
        
    except Exception as e:
        print(f"Error generating suggestions with agent: {e}")
        # Return fallback suggestions
        return [
            {
                "title": "More Details",
                "message": "",
                "partial": False,
                "className": "suggestion-item"
            },
            {
                "title": "Related Information",
                "message": "",
                "partial": False,
                "className": "suggestion-item"
            }
        ]

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint to process a query and return the response.
    """
    try:
        # Get the query from the request
        data = request.json
        question = data.get('query', '')
        session_id = data.get('session_id', f"session_{int(time.time())}")
        callback_url = data.get('callback_url', None)  # Optional callback URL for frontend

        if not question:
            return jsonify({"error": "Query parameter is required"}), 400

        # Store callback URL if provided
        if callback_url:
            suggestion_callbacks[session_id] = callback_url

        # Use the function from general_rag_demo.py to process the query
        response = initialize_and_process_query(question, kg_name, model_name)
        print(f"Kg Name from the API : {kg_name}")
        
        # Add session_id to response for frontend reference
        response['session_id'] = session_id
        
        # Trigger suggestion generation in background (async)
        try:
            def trigger_suggestion_generation():
                try:
                    suggestion_payload = {
                        "question": question,
                        "response": response,
                        "session_id": session_id
                    }
                    # Call the suggestion generation endpoint
                    requests.post(
                        "http://localhost:5000/generate-suggestions", 
                        json=suggestion_payload, 
                        timeout=30
                    )
                    print("🚀 Background suggestion generation completed")
                except Exception as e:
                    print(f"❌ Background suggestion generation failed: {e}")
            
            # Start background thread to trigger suggestion generation
            threading.Thread(target=trigger_suggestion_generation, daemon=True).start()
            print("🔄 Started background suggestion generation")
            
        except Exception as suggestion_error:
            print(f"⚠️ Warning: Could not start suggestion generation: {suggestion_error}")

        # Return the response as JSON (WITHOUT suggestions)
        return jsonify(response)

    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-suggestions', methods=['POST'])
def generate_suggestions():
    """
    API endpoint to generate suggestions based on a question and response.
    This endpoint is called separately after the main chat response.
    """
    try:
        print("🎯 Suggestion generation endpoint triggered!")
        
        # Get the data from the request
        data = request.json
        question = data.get('question', '')
        response_data = data.get('response', {})
        session_id = data.get('session_id', 'unknown')

        if not question:
            return jsonify({"error": "Question parameter is required"}), 400

        print(f"📝 Generating suggestions for session {session_id}, question: {question[:50]}...")
        
        # Generate suggestions using the agno Agent with Claude
        suggestions = generate_suggestions_with_agent(question, response_data)
        
        print(f"✅ Generated {len(suggestions)} suggestions successfully for session {session_id}")
        
        # Try to send suggestions to callback URL if provided
        if session_id in suggestion_callbacks:
            try:
                callback_url = suggestion_callbacks[session_id]
                print(f"📤 Sending suggestions to frontend callback: {callback_url}")
                
                # Prepare comprehensive payload with question and full response
                callback_payload = {
                    "type": "suggestions_ready",
                    "session_id": session_id,
                    "original_question": question,
                    "original_response": response_data,  # Full response from main chat API
                    "suggestions": suggestions,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "suggestion_count": len(suggestions),
                        "generation_source": "claude_agent",
                        "response_brief": response_data.get('brief_answer', 'N/A')
                    }
                }
                
                # Send to frontend callback URL
                callback_response = requests.post(
                    callback_url, 
                    json=callback_payload, 
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                
                if callback_response.status_code == 200:
                    print(f"✅ Successfully sent suggestions to frontend (status: {callback_response.status_code})")
                else:
                    print(f"⚠️ Frontend callback returned status: {callback_response.status_code}")
                
                # Clean up callback URL after use
                del suggestion_callbacks[session_id]
                
            except requests.exceptions.RequestException as callback_error:
                print(f"❌ Network error sending suggestions to frontend: {callback_error}")
            except Exception as callback_error:
                print(f"❌ Failed to send suggestions to frontend: {callback_error}")
        
        return jsonify({"suggestions": suggestions, "session_id": session_id})

    except Exception as e:
        print(f"❌ Error generating suggestions: {e}")
        return jsonify({"error": str(e), "suggestions": []}), 500

@app.route('/api/test-callback', methods=['POST'])
def test_callback():
    """
    Test endpoint that can be used as a callback URL for testing.
    Your frontend can use this URL to see how suggestions are delivered.
    """
    try:
        data = request.json
        print("\n" + "="*60)
        print("🎯 CALLBACK RECEIVED - SUGGESTIONS READY!")
        print("="*60)
        print(f"📧 Type: {data.get('type')}")
        print(f"🆔 Session ID: {data.get('session_id')}")
        print(f"❓ Original Question: {data.get('original_question')}")
        print(f"💬 Original Response Brief: {data.get('metadata', {}).get('response_brief')}")
        print(f"📊 Number of Suggestions: {len(data.get('suggestions', []))}")
        print(f"⏰ Timestamp: {data.get('timestamp')}")
        print("\n📝 Generated Suggestions:")
        
        for i, suggestion in enumerate(data.get('suggestions', []), 1):
            print(f"  {i}. {suggestion.get('title')} - {suggestion.get('message')}")
        
        print("="*60)
        print("✅ Callback processed successfully!")
        print("="*60 + "\n")
        
        return jsonify({
            "status": "success",
            "message": "Callback received and processed",
            "received_suggestions": len(data.get('suggestions', []))
        })
        
    except Exception as e:
        print(f"❌ Error in test callback: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced GraphRAG Demo API')
    parser.add_argument('--model', type=str, default="gemini-2.5-pro",
                        help='LLM to use: gemini-2.5-pro (auto-detects Vertex AI), vertex_ai/gemini-2.5-pro, anthropic/claude-3-sonnet-20240229')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the API server on (default: 5000)')
    parser.add_argument('--use-kg', type=str, default="default_graph",
                        help='Use an existing knowledge graph by name (default: default_graph)')
    
    args = parser.parse_args()
    model_name = args.model
    kg_name = args.use_kg
    
    print(f"Starting API server with model: {model_name}")
    print("Claude agent will be used for suggestion generation")
    
    app.run(debug=True, port=args.port)