#!/usr/bin/env bash
# Kurzname für die volle Verifikation — siehe scripts/verify.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/verify.sh"
