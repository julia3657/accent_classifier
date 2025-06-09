from flask import Flask, request, jsonify,  render_template
from flask_cors import CORS
import os
import traceback
import logging
from video_to_audio_tool.video_to_audio_tool import video_to_audio_tool_main_flow
from classifier_main import run_analysis_with_choice  # Correct analysis function

app = Flask(__name__)
CORS(app)

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') 

@app.route('/full-analysis', methods=['POST'])
def full_analysis():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Missing request data'}), 400

        if 'url' not in data:
            return jsonify({'success': False, 'error': 'Missing url'}), 400

        video_url = data['url']

        logger.info(f"Starting full analysis for: {video_url}")

        # Extract audio from video URL
        audio_result = video_to_audio_tool_main_flow(video_url)

        if isinstance(audio_result, dict):
            if not audio_result.get('success'):
                return jsonify({
                    'success': False,
                    'error': f"Audio extraction failed: {audio_result.get('error', 'Unknown')}",
                    'step': 'audio_extraction'
                }), 500
            audio_path = audio_result.get('audio_path')
        else:
            return jsonify({'success': False, 'error': 'Invalid audio result format'}), 500

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'success': False, 'error': 'Audio file not created'}), 500

        logger.info("Audio extracted. Proceeding to accent analysis...")

        analysis_result = run_analysis_with_choice(audio_path=audio_path)
        if analysis_result is not None:
            analysis_result.update({
                'success': True,
                'video_url': video_url,
                'audio_path': audio_path
            })
        return jsonify(analysis_result)

    except Exception as e:
        logger.error(f"Error in full_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
