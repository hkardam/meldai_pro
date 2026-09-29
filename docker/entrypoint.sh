#!/usr/bin/env bash
set -e

echo "================================================================="
echo "  🩺 MeldAI Medical Science & Clinical Intelligence Engine"
echo "================================================================="
echo "  Environment     : ${ENVIRONMENT:-development}"
echo "  Debug Mode      : ${DEBUG_MODE:-false} (Port: ${DEBUG_PORT:-5678})"
echo "  Wait Debug      : ${WAIT_FOR_DEBUGGER:-false}"
echo "  PostgreSQL Host : ${POSTGRES_HOST:-host.docker.internal}:${POSTGRES_PORT:-5432} (External)"
echo "  MongoDB Host    : ${MONGO_HOST:-mongodb}:${MONGO_PORT:-27017} (Internal)"
echo "================================================================="

# Debugpy wrapper logic
if [ "${DEBUG_MODE}" = "true" ]; then
    PORT="${DEBUG_PORT:-5678}"
    echo "🚀 [DEBUGGER] Debugpy is listening on 0.0.0.0:${PORT}"
    echo "💡 Attach your IDE (VS Code 'Python: Attach to Docker Container') to localhost:${PORT}"
    
    DEBUG_ARGS="-m debugpy --listen 0.0.0.0:${PORT}"
    
    if [ "${WAIT_FOR_DEBUGGER}" = "true" ]; then
        echo "⏳ [WAIT] Waiting for IDE debugger to attach before starting code execution..."
        DEBUG_ARGS="${DEBUG_ARGS} --wait-for-client"
    fi

    # If the user passed python command, inject debugpy
    if [ "$1" = "python" ]; then
        shift
        exec python ${DEBUG_ARGS} "$@"
    elif [ "$1" = "meldai" ]; then
        shift
        exec python ${DEBUG_ARGS} -m meldai.main "$@"
    fi
fi

# Fallback: execute any arbitrary command (e.g., bash, pytest, etc.)
exec "$@"
