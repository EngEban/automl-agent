# ============================================================
# scaffold.ps1 - Create full professional project tree
# Run from: C:\Eban_projects\autodata-tool
# Idempotent: safe to run multiple times
# Does NOT overwrite existing files with content
# ============================================================
$ErrorActionPreference = "Stop"

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
        Write-Host "  [dir]  $path" -ForegroundColor Green
    }
}

function Ensure-File($path) {
    if (-not (Test-Path $path)) {
        $dir = Split-Path $path -Parent
        if ($dir -and -not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
        New-Item -ItemType File -Force -Path $path | Out-Null
        Write-Host "  [file] $path" -ForegroundColor Cyan
    }
}

# ============================================================
# STEP 1 - Migrate existing files
# ============================================================
Write-Host ""
Write-Host "== Step 1: Migrate existing files ==" -ForegroundColor Yellow

# routes.py -> routes/analysis.py
if ((Test-Path "backend\app\api\routes.py") -and -not (Test-Path "backend\app\api\routes\analysis.py")) {
    Ensure-Dir "backend\app\api\routes"
    Move-Item "backend\app\api\routes.py" "backend\app\api\routes\analysis.py"
    Write-Host "  [move] routes.py -> routes/analysis.py" -ForegroundColor Magenta
}

# schemas/__init__.py -> schemas/analysis.py (only if substantive)
if ((Test-Path "backend\app\schemas\__init__.py") -and -not (Test-Path "backend\app\schemas\analysis.py")) {
    $size = (Get-Item "backend\app\schemas\__init__.py").Length
    if ($size -gt 300) {
        Move-Item "backend\app\schemas\__init__.py" "backend\app\schemas\analysis.py"
        Write-Host "  [move] schemas/__init__.py -> schemas/analysis.py" -ForegroundColor Magenta
    }
}

# ============================================================
# STEP 2 - Create directories
# ============================================================
Write-Host ""
Write-Host "== Step 2: Directory tree ==" -ForegroundColor Yellow

$dirs = @(
    "backend\app\api\routes",
    "backend\app\agent",
    "backend\app\db",
    "backend\app\services",
    "backend\app\utils",
    "backend\tests",
    "backend\alembic\versions",
    "backend\storage\uploads",
    "backend\storage\reports",
    "backend\storage\figures",
    "backend\storage\models",
    "frontend\src",
    "docker",
    "docs"
)
foreach ($d in $dirs) { Ensure-Dir $d }

# ============================================================
# STEP 3 - Create files (empty)
# ============================================================
Write-Host ""
Write-Host "== Step 3: Files ==" -ForegroundColor Yellow

# Backend root
Ensure-File "backend\alembic.ini"

# Alembic
Ensure-File "backend\alembic\env.py"
Ensure-File "backend\alembic\script.py.mako"
Ensure-File "backend\alembic\versions\.gitkeep"

# API
Ensure-File "backend\app\api\__init__.py"
Ensure-File "backend\app\api\deps.py"
Ensure-File "backend\app\api\router.py"
Ensure-File "backend\app\api\routes\__init__.py"
Ensure-File "backend\app\api\routes\analysis.py"
Ensure-File "backend\app\api\routes\sessions.py"
Ensure-File "backend\app\api\routes\chat.py"
Ensure-File "backend\app\api\routes\artifacts.py"
Ensure-File "backend\app\api\routes\health.py"

# Agent
Ensure-File "backend\app\agent\__init__.py"
Ensure-File "backend\app\agent\llm.py"
Ensure-File "backend\app\agent\prompts.py"
Ensure-File "backend\app\agent\tools.py"
Ensure-File "backend\app\agent\agent.py"
Ensure-File "backend\app\agent\memory.py"
Ensure-File "backend\app\agent\executor.py"

# Core (exists - will not be touched)
Ensure-File "backend\app\core\__init__.py"

# DB
Ensure-File "backend\app\db\__init__.py"
Ensure-File "backend\app\db\base.py"
Ensure-File "backend\app\db\models.py"
Ensure-File "backend\app\db\session.py"
Ensure-File "backend\app\db\crud.py"

# Schemas
Ensure-File "backend\app\schemas\__init__.py"
Ensure-File "backend\app\schemas\analysis.py"
Ensure-File "backend\app\schemas\session.py"
Ensure-File "backend\app\schemas\chat.py"
Ensure-File "backend\app\schemas\agent.py"
Ensure-File "backend\app\schemas\common.py"

# Services
Ensure-File "backend\app\services\__init__.py"
Ensure-File "backend\app\services\session_service.py"
Ensure-File "backend\app\services\analysis_service.py"
Ensure-File "backend\app\services\pipeline_service.py"

# Tasks
Ensure-File "backend\app\tasks\__init__.py"
Ensure-File "backend\app\tasks\agent_tasks.py"

# Utils
Ensure-File "backend\app\utils\__init__.py"
Ensure-File "backend\app\utils\files.py"
Ensure-File "backend\app\utils\logging.py"
Ensure-File "backend\app\utils\validators.py"

# Tests
Ensure-File "backend\tests\__init__.py"
Ensure-File "backend\tests\conftest.py"
Ensure-File "backend\tests\test_data_loader.py"
Ensure-File "backend\tests\test_preprocessing.py"
Ensure-File "backend\tests\test_trainer.py"
Ensure-File "backend\tests\test_eda.py"
Ensure-File "backend\tests\test_agent.py"
Ensure-File "backend\tests\test_api.py"

# Root files
Ensure-File "LICENSE"
Ensure-File "pyproject.toml"
Ensure-File "Makefile"

# Docs
Ensure-File "docs\ARCHITECTURE.md"
Ensure-File "docs\API.md"
Ensure-File "docs\SETUP.md"
Ensure-File "docs\AGENT.md"
Ensure-File "docs\CONTRIBUTING.md"

# Frontend / Docker placeholders
Ensure-File "frontend\.gitkeep"
Ensure-File "docker\.gitkeep"

# ============================================================
# STEP 4 - Restore minimal re-exports (so imports don't break)
# ============================================================
Write-Host ""
Write-Host "== Step 4: Minimal re-exports (safety) ==" -ForegroundColor Yellow
$enc = New-Object System.Text.UTF8Encoding($false)

# routes/__init__.py
$routesInit = @'
"""
Routes package. Re-exports for backward compatibility.
"""
from app.api.routes.analysis import router

__all__ = ["router"]
'@
if ((Get-Item "backend\app\api\routes\__init__.py").Length -eq 0) {
    [System.IO.File]::WriteAllText("$PWD\backend\app\api\routes\__init__.py", $routesInit, $enc)
    Write-Host "  [init] routes/__init__.py (re-export)" -ForegroundColor Cyan
}

# schemas/__init__.py
$schemasInit = @'
"""
Schemas package. Re-exports for backward compatibility.
"""
from app.schemas.analysis import (
    AnalyzeURLRequest,
    TaskResponse,
    TaskStatus,
)

__all__ = [
    "AnalyzeURLRequest",
    "TaskResponse",
    "TaskStatus",
]
'@
if ((Get-Item "backend\app\schemas\__init__.py").Length -eq 0) {
    [System.IO.File]::WriteAllText("$PWD\backend\app\schemas\__init__.py", $schemasInit, $enc)
    Write-Host "  [init] schemas/__init__.py (re-export)" -ForegroundColor Cyan
}

# ============================================================
# STEP 5 - Summary
# ============================================================
Write-Host ""
Write-Host "== Summary ==" -ForegroundColor Yellow
$pyFiles = (Get-ChildItem backend -Recurse -File -Filter *.py).Count
$allFiles = (Get-ChildItem -Recurse -File).Count
$allDirs = (Get-ChildItem -Recurse -Directory).Count
Write-Host "  Python files : $pyFiles" -ForegroundColor Cyan
Write-Host "  Total files  : $allFiles" -ForegroundColor Cyan
Write-Host "  Total dirs   : $allDirs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Scaffold complete." -ForegroundColor Green