from flask import Blueprint, jsonify, request

from ..services.transcript_service import list_transcript_files_gcp

transcripts_bp = Blueprint("transcripts", __name__)


@transcripts_bp.route("/transcripts", methods=["GET"])
def list_transcripts():
    """
    List transcript files from GCP bucket with pagination and optional search.
    This route is minimal; it delegates storage and metadata parsing to the transcript service.
    """
    try:
        limit_param = request.args.get("limit", "9")
        offset_param = request.args.get("offset", "0")
        search_param = request.args.get("search") or request.args.get("q")

        try:
            limit = int(limit_param) if limit_param else 9
        except (ValueError, TypeError):
            limit = 9

        try:
            offset = int(offset_param) if offset_param else 0
        except (ValueError, TypeError):
            offset = 0

        if limit < 1:
            limit = 9
        if offset < 0:
            offset = 0

        transcripts, total_count = list_transcript_files_gcp(limit=limit, offset=offset, search=search_param)
        return jsonify(
            {
                "transcripts": transcripts,
                "totalCount": total_count,
                "limit": limit,
                "offset": offset,
            }
        )
    except Exception as exc:  # pragma: no cover - guardrail for unexpected storage failures
        return jsonify({"error": str(exc)}), 500
