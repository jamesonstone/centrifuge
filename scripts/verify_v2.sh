#!/bin/bash

# Verification script for V2 migration
set -e

echo "🔍 Verifying V2 Migration"
echo "========================="
echo ""

# Check files exist
echo "✓ Checking core files..."
FILES=(
    "api/main.py"
    "worker/main.py"
    "core/database.py"
    "core/pipeline.py"
    "core/storage.py"
    "core/llm_client.py"
    "Makefile"
    "docs/RUNBOOK.md"
    "docs/MIGRATION_V1_TO_V2.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing!"
        exit 1
    fi
done

echo ""
echo "✓ Checking Python imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from core.database import DatabasePool, RunManager
    print('  ✓ core.database imports correctly')
except ImportError as e:
    print(f'  ✗ core.database import failed: {e}')
    sys.exit(1)

try:
    from core.pipeline import DataPipeline
    print('  ✓ core.pipeline imports correctly')
except ImportError as e:
    print(f'  ✗ core.pipeline import failed: {e}')
    sys.exit(1)

try:
    from core.storage import StorageBackend
    print('  ✓ core.storage imports correctly')
except ImportError as e:
    print(f'  ✗ core.storage import failed: {e}')
    sys.exit(1)

try:
    from core.llm_client import LLMClient
    print('  ✓ core.llm_client imports correctly')
except ImportError as e:
    print(f'  ✗ core.llm_client import failed: {e}')
    sys.exit(1)
"

echo ""
echo "✓ Checking API references..."
grep -q "api.main:app" api/main.py && echo "  ✓ API self-reference correct"
grep -q "api.main:app" docker-compose.yaml && echo "  ✓ Docker-compose reference correct"
grep -q "api.main.py" Makefile && echo "  ✓ Makefile reference correct"

echo ""
echo "✓ Checking backup files..."
if [ -f "api/main_v1_backup.py" ]; then
    echo "  ℹ️  V1 API backup exists (can be deleted if migration successful)"
fi
if [ -f "Makefile.v1_backup" ]; then
    echo "  ℹ️  V1 Makefile backup exists (can be deleted if migration successful)"
fi

echo ""
echo "✅ V2 Migration Verification Complete!"
echo ""
echo "Next steps:"
echo "1. Start the stack: make stack-up"
echo "2. Check health: make health-check"
echo "3. Run tests: make test-all"
echo "4. If everything works, delete backups:"
echo "   rm api/main_v1_backup.py Makefile.v1_backup"