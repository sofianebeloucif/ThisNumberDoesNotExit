from flask import Flask, render_template, jsonify, request
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Add the src folder to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from generator import GlobalGenerator, ConditionalGenerator

app = Flask(__name__)

# Load models at startup
print("Loading models...")

try:
    global_gen = GlobalGenerator.load('../models/global_generator.pkl')
    has_global = True
except FileNotFoundError:
    print("⚠️  Global model not found")
    has_global = False

try:
    cond_gen = ConditionalGenerator.load('../models/conditional_generator.pkl')
    has_conditional = True
except FileNotFoundError:
    print("⚠️  Conditional model not found")
    has_conditional = False

if not has_global and not has_conditional:
    print("❌ No model found! Please run train_and_compare.ipynb first")
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
    """Home page"""
    return render_template('index.html',
                           has_global=has_global,
                           has_conditional=has_conditional)


@app.route('/generate', methods=['POST'])
def generate():
    """Endpoint to generate images"""
    try:
        data = request.get_json()
        n_samples = int(data.get('n_samples', 1))
        use_rejection = data.get('use_rejection', True)
        percentile = int(data.get('percentile', 25))
        mode = data.get('mode', 'global')
        digit = data.get('digit', None)
        clean_images = data.get('clean_images', True)
        cleaning_method = data.get('cleaning_method', 'medium')

        # Limit the number of samples
        n_samples = min(max(1, n_samples), 16)

        # Validate percentile
        if percentile not in [10, 25, 50]:
            percentile = 25

        # Validate cleaning method
        if cleaning_method not in ['light', 'medium', 'aggressive']:
            cleaning_method = 'medium'

        # Generate according to mode
        if mode == 'conditional' and has_conditional:
            if digit is None:
                return jsonify({
                    'success': False,
                    'error': 'Digit is required in conditional mode'
                }), 400

            digit = int(digit)
            if digit < 0 or digit > 9:
                return jsonify({
                    'success': False,
                    'error': 'Digit must be between 0 and 9'
                }), 400

            images = cond_gen.generate(digit, n_samples, use_rejection, percentile,
                                       clean_images, cleaning_method)

        elif mode == 'global' and has_global:
            images = global_gen.generate(n_samples, use_rejection, percentile,
                                         clean_images, cleaning_method)

        else:
            return jsonify({
                'success': False,
                'error': f'Mode {mode} not available'
            }), 400

        # Convert to base64
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats')
def stats():
    """Return statistics about the models"""
    stats_data = {
        'available_modes': []
    }

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
    app.run(debug=True, host='0.0.0.0', port=5000)