import os
import sys
from defender.apps import create_app
from defender.models.whitebox_mlp_model import WhiteboxMLPEmberModel

HERE = os.path.dirname(__file__)
def abspath(p): return p if os.path.isabs(p) else os.path.join(HERE, p)

def build_app():
    # default relative to this file
    default_weights = "models/whitebox_mlp.pt"
    cfg = os.getenv("DF_MODEL_GZ_PATH", default_weights)
    model_path = abspath(cfg)  # << no extra "models/" prefixing here
    
    print(f"Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        # Try alternative paths
        alt_paths = [
            os.path.join(HERE, "..", "models", "whitebox_mlp.pt"),
            "/opt/defender/defender/models/whitebox_mlp.pt"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Found model at alternative path: {alt_path}")
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find model file. Checked: {model_path}, {alt_paths}")

    thr = os.getenv("DF_MODEL_THRESH")
    threshold = None
    if thr:
        try:
            threshold = float(thr)
        except ValueError:
            print(f"Warning: Invalid threshold value '{thr}', using default")
    
    print(f"Initializing model with threshold: {threshold}")
    model = WhiteboxMLPEmberModel(
        model_gz_path=model_path,
        model_thresh=threshold,
        model_name=os.getenv("DF_MODEL_NAME", "whitebox_mlp"),
    )
    return create_app(model), model_path

def main():
    try:
        print("Starting defender application...")
        app, model_path = build_app()
        print(f"Model loaded from: {model_path}")
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
