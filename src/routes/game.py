from flask import Blueprint, render_template, request, jsonify
from src.database import db
import os
import logging

game_bp = Blueprint('game', __name__)
logger = logging.getLogger(__name__)

@game_bp.route('/game/snake')
def snake_game():
    liff_id = os.getenv('LIFF_ID')
    return render_template('snake.html', liff_id=liff_id)

@game_bp.route('/api/score', methods=['POST'])
def submit_score():
    if db is None:
         return jsonify({'status': 'error', 'message': 'Database not initialized'}), 503

    data = request.json
    user_id = data.get('user_id')
    score = data.get('score')

    if not user_id or score is None:
        return jsonify({'status': 'error', 'message': 'Missing user_id or score'}), 400

    try:
        success = db.add_game_score(user_id, int(score))
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Database error'}), 500
    except Exception as e:
        logger.error(f"Error submitting score: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@game_bp.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    if db is None:
         return jsonify({'status': 'error', 'message': 'Database not initialized'}), 503

    try:
        limit = request.args.get('limit', 10, type=int)
        scores = db.get_top_scores(limit=limit)
        return jsonify(scores)
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
