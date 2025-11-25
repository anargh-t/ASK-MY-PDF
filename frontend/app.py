import os
import base64
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import requests
from datetime import datetime
import uuid
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max file size

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize session defaults
@app.before_request
def init_session():
    if 'doc_id' not in session:
        session['doc_id'] = None
    if 'extracted' not in session:
        session['extracted'] = False
    if 'messages' not in session:
        session['messages'] = []
    if 'filename' not in session:
        session['filename'] = None
    if 'history' not in session:
        session['history'] = []
    if 'last_metrics' not in session:
        session['last_metrics'] = {
            'latency_ms': 0,
            'retrieved': 0,
            'retrieval_accuracy': 0,
            'relevance_score': 0
        }


@app.route('/')
def index():
    """Landing page"""
    return render_template('landing.html')


@app.route('/chat')
def chat():
    """Main chat interface"""
    # Check backend status
    backend_status = {'connected': False, 'documents': 0, 'history': 0}
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        if response.status_code == 200:
            try:
                backend_status = response.json()
                backend_status['connected'] = True
            except ValueError:
                backend_status['connected'] = False
    except requests.exceptions.ConnectionError:
        backend_status['connected'] = False
        backend_status['error'] = 'Cannot connect to backend'
    except Exception:
        backend_status['connected'] = False
    
    # Get PDF URL if doc_id exists
    pdf_url = None
    if session.get('doc_id'):
        pdf_url = f"/api/pdf/{session.get('doc_id')}"
    
    return render_template('index.html', 
                         backend_status=backend_status,
                         doc_id=session.get('doc_id'),
                         extracted=session.get('extracted', False),
                         filename=session.get('filename'),
                         messages=session.get('messages', []),
                         history=session.get('history', []),
                         metrics=session.get('last_metrics', {}),
                         pdf_url=pdf_url,
                         backend_url=BACKEND_URL)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Read file content
        file_content = file.read()
        file.seek(0)  # Reset file pointer for potential reuse
        
        # Upload to backend
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files={'file': (file.filename, file_content, 'application/pdf')},
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                data = response.json()
                session['doc_id'] = data['doc_id']
                session['filename'] = data['filename']
                session['extracted'] = False
                session['messages'] = []
                return jsonify(data)
            except ValueError:
                return jsonify({'error': 'Invalid response from backend'}), 500
        else:
            try:
                error = response.json().get('detail', 'Upload failed')
            except ValueError:
                error = f'Upload failed with status {response.status_code}'
            return jsonify({'error': error}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Cannot connect to backend. Make sure it is running on port 8000'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Backend request timed out'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Backend connection error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500


@app.route('/api/extract', methods=['POST'])
def extract_text():
    """Extract and index PDF text"""
    data = request.json
    doc_id = data.get('doc_id') or session.get('doc_id')
    
    if not doc_id:
        return jsonify({'error': 'No document uploaded'}), 400
    
    chunk_size = data.get('chunk_size', 500)
    overlap = data.get('overlap', 100)
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/extract-text",
            json={
                'doc_id': doc_id,
                'chunk_size': chunk_size,
                'overlap': overlap
            },
            timeout=120
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                session['extracted'] = True
                return jsonify(result)
            except ValueError:
                return jsonify({'error': 'Invalid response from backend'}), 500
        else:
            try:
                error = response.json().get('detail', 'Extraction failed')
            except ValueError:
                error = f'Extraction failed with status {response.status_code}'
            return jsonify({'error': error}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Cannot connect to backend. Make sure it is running on port 8000'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Backend request timed out'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Backend connection error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Extraction error: {str(e)}'}), 500


@app.route('/api/query', methods=['POST'])
def query_pdf():
    """Query the PDF"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.json or {}
    doc_id = data.get('doc_id') or session.get('doc_id')
    question = data.get('question', '').strip()
    
    if not doc_id:
        return jsonify({'error': 'No document uploaded'}), 400
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    # Check if extracted - but allow query even if session says not extracted
    # (backend will auto-index if needed)
    top_k = data.get('top_k', 5)
    relevance_score = data.get('relevance_score')
    
    try:
        # Log request details
        print(f"Query request - doc_id: {doc_id}, question: {question[:50]}..., top_k: {top_k}")
        
        # Get conversation history from session (last 10 messages for context)
        conversation_history = []
        if 'messages' in session and session['messages']:
            # Get last 10 messages (5 exchanges) for context
            recent_messages = session['messages'][-10:]
            conversation_history = [
                {'role': msg.get('role', 'user'), 'content': msg.get('content', '')}
                for msg in recent_messages
            ]
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                'doc_id': doc_id,
                'question': question,
                'top_k': top_k,
                'relevance_score': relevance_score,
                'conversation_history': conversation_history if conversation_history else None
            },
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        
        # Log response for debugging
        print(f"Backend response status: {response.status_code}")
        print(f"Backend response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Backend response data keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                
                # Mark as extracted if query succeeds
                session['extracted'] = True
                session['doc_id'] = doc_id
                
                # Add messages to session
                if 'messages' not in session:
                    session['messages'] = []
                
                session['messages'].append({
                    'role': 'user',
                    'content': question,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                session['messages'].append({
                    'role': 'assistant',
                    'content': result.get('answer', ''),
                    'references': result.get('references', []),
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Update metrics
                if 'metrics' in result:
                    session['last_metrics'] = result['metrics']
                
                # Ensure result includes doc_id
                result['doc_id'] = doc_id
                
                return jsonify(result)
            except ValueError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response text: {response.text[:500]}")
                return jsonify({'error': f'Invalid response from backend: {str(e)}'}), 500
        else:
            try:
                error_data = response.json()
                error = error_data.get('detail', 'Query failed')
                print(f"Backend error (JSON): {error}")
            except ValueError:
                error_text = response.text[:500] if response.text else 'No response body'
                error = f'Query failed with status {response.status_code}. Response: {error_text}'
                print(f"Backend error (non-JSON): {error}")

            # If backend lost the document (e.g., after restart) reset session
            if response.status_code == 404 and "Document not found" in error:
                session['doc_id'] = None
                session['extracted'] = False
                session['messages'] = []
                session['filename'] = None
                session['last_metrics'] = {
                    'latency_ms': 0,
                    'retrieved': 0,
                    'retrieval_accuracy': 0,
                    'relevance_score': 0
                }
                error = "Document not found on server. Please re-upload and extract your PDF."

            return jsonify({'error': error}), response.status_code
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return jsonify({'error': 'Cannot connect to backend. Make sure it is running on port 8000'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Backend request timed out'}), 504
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return jsonify({'error': f'Backend connection error: {str(e)}'}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Query error: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get query history from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/history", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({'history': []})
    except:
        return jsonify({'history': []})


@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Clear current session"""
    session['messages'] = []
    session['last_metrics'] = {
        'latency_ms': 0,
        'retrieved': 0,
        'retrieval_accuracy': 0,
        'relevance_score': 0
    }
    return jsonify({'success': True})


@app.route('/api/new-session', methods=['POST'])
def new_session():
    """Save current session and start new one"""
    # Save to history if there's content
    if session.get('messages') or session.get('doc_id'):
        history_entry = {
            'id': str(uuid.uuid4()),
            'name': session.get('filename') or f'Session {len(session.get("history", [])) + 1}',
            'doc_id': session.get('doc_id'),
            'filename': session.get('filename'),
            'messages': session.get('messages', []).copy(),
            'created_at': datetime.utcnow().isoformat()
        }
        if 'history' not in session:
            session['history'] = []
        session['history'].insert(0, history_entry)
    
    # Clear current session
    session['doc_id'] = None
    session['extracted'] = False
    session['messages'] = []
    session['filename'] = None
    
    return jsonify({'success': True, 'history': session.get('history', [])})


@app.route('/api/load-session', methods=['POST'])
def load_session():
    """Load a saved session"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or 'history' not in session:
        return jsonify({'error': 'Session not found'}), 404
    
    # Find session in history
    history = session.get('history', [])
    saved_session = next((s for s in history if s.get('id') == session_id), None)
    
    if not saved_session:
        return jsonify({'error': 'Session not found'}), 404
    
    # Restore session
    session['doc_id'] = saved_session.get('doc_id')
    session['filename'] = saved_session.get('filename')
    session['messages'] = saved_session.get('messages', []).copy()
    session['extracted'] = True  # Assume extracted if it was saved
    
    return jsonify({'success': True})


@app.route('/api/pdf/<doc_id>')
def get_pdf(doc_id):
    """Serve PDF file by doc_id"""
    try:
        # Find PDF file in upload directory
        # Backend stores files as {uuid}_{filename}, so we need to search
        upload_dir = Path(__file__).resolve().parent.parent / "data" / "uploaded_files"
        
        # Get filename from session if available
        filename = session.get('filename')
        if filename:
            # Try to find file with this filename (most recent match)
            pdf_files = sorted(upload_dir.glob(f"*_{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pdf_files:
                return send_file(pdf_files[0], mimetype='application/pdf')
        
        # Fallback: try to find any PDF (this is less precise but works)
        pdf_files = sorted(upload_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pdf_files:
            # Return most recent PDF as fallback
            return send_file(pdf_files[0], mimetype='application/pdf')
        
        return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
