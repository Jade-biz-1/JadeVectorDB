#!/usr/bin/env python3
"""
Script to split tasks.md into organized files in TasksTracking folder
"""
import re
from pathlib import Path

# Define the split configuration
SPLITS = {
    "01-setup-foundational.md": {
        "title": "Setup & Foundational Infrastructure",
        "phases": ["Phase 1: Setup", "Phase 2: Foundational"],
        "task_range": "T001-T027",
        "phase_range": "1-2",
        "status": "100% Complete ✅"
    },
    "02-core-features.md": {
        "title": "Core Vector Database Features",
        "phases": [
            "Phase 3: User Story 1 - Vector Storage and Retrieval",
            "Phase 4: User Story 2 - Similarity Search",
            "Phase 5: User Story 3 - Advanced Similarity Search with Filters",
            "Phase 6: User Story 4 - Database Creation and Configuration"
        ],
        "task_range": "T028-T087",
        "phase_range": "3-6",
        "status": "100% Complete ✅"
    },
    "03-advanced-features.md": {
        "title": "Advanced Features & Capabilities",
        "phases": [
            "Phase 7: User Story 5 - Embedding Management",
            "Phase 8: User Story 6 - Distributed Deployment and Scaling",
            "Phase 9: User Story 7 - Vector Index Management",
            "Phase 10: User Story 9 - Vector Data Lifecycle Management"
        ],
        "task_range": "T088-T162",
        "phase_range": "7-10",
        "status": "100% Complete ✅"
    },
    "04-monitoring-polish.md": {
        "title": "Monitoring & Cross-Cutting Concerns",
        "phases": [
            "Phase 11: User Story 8 - Monitoring and Health Status",
            "Phase 12: Polish & Cross-Cutting Concerns"
        ],
        "task_range": "T163-T214",
        "phase_range": "11-12",
        "status": "100% Complete ✅"
    },
    "05-tutorial.md": {
        "title": "Interactive Tutorial Development",
        "phases": [
            "Phase 13: Interactive Tutorial Development"
        ],
        "task_range": "T215.01-T218",
        "phase_range": "13",
        "status": "83% Complete 🔄"
    },
    "06-current-auth-api.md": {
        "title": "Authentication & API Completion (CURRENT FOCUS)",
        "phases": [
            "Phase 14: Next Session Focus - Authentication & API Completion"
        ],
        "task_range": "T219-T238",
        "phase_range": "14",
        "status": "60% Complete 🔄"
    }
}

def read_tasks_file(filepath):
    """Read the tasks.md file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_phase_content(content, phase_title):
    """Extract content for a specific phase"""
    # Find the phase header
    pattern = rf"^## {re.escape(phase_title)}.*?$"
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return None

    start = match.start()

    # Find the next phase header (## Phase or ##Task Status or end of file)
    next_phase = re.search(r'^## (?:Phase|Task Status)', content[start+1:], re.MULTILINE)

    if next_phase:
        end = start + next_phase.start()
        return content[start:end].strip()
    else:
        return content[start:].strip()

def create_split_file(output_dir, filename, config, phase_contents):
    """Create a split task file"""
    filepath = output_dir / filename

    # Build the file content
    lines = []
    lines.append(f"# {config['title']}\n")
    lines.append(f"**Phase**: {config['phase_range']}")
    lines.append(f"**Task Range**: {config['task_range']}")
    lines.append(f"**Status**: {config['status']}")
    lines.append(f"**Last Updated**: 2025-12-06\n")
    lines.append("---\n")
    lines.append("## Phase Overview\n")

    phase_list = "\n".join(f"- {phase}" for phase in config['phases'])
    lines.append(f"{phase_list}\n")
    lines.append("---\n")

    # Add phase contents
    for phase_content in phase_contents:
        if phase_content:
            lines.append(f"\n{phase_content}\n")
            lines.append("---\n")

    # Write the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Created: {filepath}")

def main():
    # Paths
    tasks_file = Path("/home/deepak/Public/JadeVectorDB/specs/002-check-if-we/tasks.md")
    output_dir = Path("/home/deepak/Public/JadeVectorDB/TasksTracking")

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Read the tasks file
    print("Reading tasks.md...")
    content = read_tasks_file(tasks_file)

    # Process each split
    for filename, config in SPLITS.items():
        print(f"\nProcessing {filename}...")

        # Extract content for each phase in this split
        phase_contents = []
        for phase_title in config['phases']:
            phase_content = extract_phase_content(content, phase_title)
            if phase_content:
                phase_contents.append(phase_content)

        # Create the split file
        if phase_contents:
            create_split_file(output_dir, filename, config, phase_contents)
        else:
            print(f"  Warning: No content found for {filename}")

    print("\nDone!")

if __name__ == "__main__":
    main()
