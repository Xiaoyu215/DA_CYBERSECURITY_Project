import os
import sys
from defender.apps import create_app
from defender.models.whitebox_mlp_model import WhiteboxMLPEmberModel
from pathlib import Path

HERE = os.path.dirname(__file__)
def abspath(p): return p if os.path.isabs(p) else os.path.join(HERE, p)

MODELS_DIR = Path("/opt/defender/defender/models")
WEIGHTS    = MODELS_DIR / "whitebox_mlp.pt"
SCALER_JS  = MODELS_DIR / "whitebox_scaler_params.json" 
SCALER_PKL = MODELS_DIR / "whitebox_scaler.joblib"       
THRESH     = MODELS_DIR / "whitebox_threshold.json"
META       = MODELS_DIR / "whitebox_model_meta.json"

def build_app():
    model = WhiteboxMLPEmberModel(
        model_gz_path=str(WEIGHTS),
        model_thresh=None,               
        model_name="whitebox_mlp",
    )
    return create_app(model)

def main():
    try:
        print("Starting defender application...")
        app = build_app()
        print("Starting Flask server on 0.0.0.0:8080...")
        
        # Use gevent for production-ready WSGI server
        from gevent.pywsgi import WSGIServer
        http_server = WSGIServer(('0.0.0.0', 8080), app)
        print("Server ready to receive requests")
        http_server.serve_forever()
        
    except Exception as e:
        print(f"Error starting application: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
