import base64
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Add backend folder to path for generator import
sys.path.insert(0, os.path.dirname(__file__))
from generator import GlobalGenerator, ConditionalGenerator

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')
# Enable CORS for frontend hosted on GitHub Pages
GITHUB_PAGES_ORIGIN = "https://sofianebeloucif.github.io"

CORS(
    app,
    resources={
        r"/generate": {"origins": GITHUB_PAGES_ORIGIN},
        r"/stats": {"origins": GITHUB_PAGES_ORIGIN}
    }
)

# Absolute paths for models (robust on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
MODEL_DIR = os.path.join(BASE_DIR, 'models')
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, 'global_generator.pkl')
CONDITIONAL_MODEL_PATH = os.path.join(MODEL_DIR, 'conditional_generator.pkl')

# Load models at startup
print("Loading models...")

try:
    global_gen = GlobalGenerator.load(GLOBAL_MODEL_PATH)
    has_global = True
except FileNotFoundError:
    print(" Global model not found")
    has_global = False

try:
    cond_gen = ConditionalGenerator.load(CONDITIONAL_MODEL_PATH)
    has_conditional = True
except FileNotFoundError:
    print(" Conditional model not found")
    has_conditional = False

if not has_global and not has_conditional:
    print(" No models found! Run training notebook first.")
    sys.exit(1)

print(f"✓ Models loaded (Global: {has_global}, Conditional: {has_conditional})")


def image_to_base64(img_array):
    """Convert a numpy array to base64 for web display"""
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img = img.resize((280, 280), Image.NEAREST)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate images endpoint"""
    try:
        data = request.get_json()
        n_samples = int(data.get('n_samples', 1))
        use_rejection = data.get('use_rejection', True)
        percentile = int(data.get('percentile', 25))
        mode = data.get('mode', 'global')
        digit = data.get('digit', None)
        clean_images = data.get('clean_images', True)
        cleaning_method = data.get('cleaning_method', 'medium')

        # Clamp sample count
        n_samples = min(max(1, n_samples), 16)

        # Validate percentile
        if percentile not in [10, 25, 50]:
            percentile = 25

        # Validate cleaning method
        if cleaning_method not in ['light', 'medium', 'aggressive']:
            cleaning_method = 'medium'

        # Generate images
        if mode == 'conditional' and has_conditional:
            if digit is None:
                return jsonify({'success': False, 'error': 'Digit is required for conditional mode'}), 400
            digit = int(digit)
            if digit < 0 or digit > 9:
                return jsonify({'success': False, 'error': 'Digit must be 0-9'}), 400
            images = cond_gen.generate(digit, n_samples, use_rejection, percentile,
                                       clean_images, cleaning_method)

        elif mode == 'global' and has_global:
            images = global_gen.generate(n_samples, use_rejection, percentile,
                                         clean_images, cleaning_method)
        else:
            return jsonify({'success': False, 'error': f'Mode {mode} not available'}), 400

        images_b64 = [image_to_base64(img) for img in images]

        return jsonify({
            'success': True,
            'images': images_b64,
            'count': len(images_b64),
            'mode': mode,
            'digit': digit if mode == 'conditional' else None,
            'use_rejection': use_rejection,
            'percentile': percentile if use_rejection else None,
            'clean_images': clean_images,
            'cleaning_method': cleaning_method if clean_images else None
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Return model statistics"""
    stats_data = {'available_modes': []}

    if has_global:
        stats_data['available_modes'].append('global')
        stats_data['global'] = {
            'n_components': global_gen.n_components,
            'bandwidth': global_gen.bandwidth,
            'rejection_available': True,
            'percentiles': [10, 25, 50]
        }

    if has_conditional:
        stats_data['available_modes'].append('conditional')
        stats_data['conditional'] = {
            'n_components': cond_gen.n_components,
            'bandwidth': cond_gen.bandwidth,
            'available_digits': list(cond_gen.kde_models.keys()),
            'rejection_available': True,
            'percentiles': [10, 25, 50]
        }

    return jsonify(stats_data)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
