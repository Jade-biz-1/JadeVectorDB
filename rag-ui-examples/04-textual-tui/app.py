#!/usr/bin/env python3
"""
RAG UI Alternative #4: Terminal UI (TUI) with Textual
Beautiful terminal-based interface, no browser needed

Usage:
    pip install textual
    python app.py

Controls:
    - Tab: Switch between input fields
    - Ctrl+Q: Quit
    - Enter: Submit query
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, TextArea, Static, Select, Button, ProgressBar
from textual.binding import Binding
from datetime import datetime
import time

class RAGApp(App):
    """Maintenance Documentation Q&A Terminal UI"""

    CSS = """
    Screen {
        background: $surface;
    }

    #title {
        background: $primary;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    .panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    #query-panel {
        height: 50%;
    }

    #result-panel {
        height: 50%;
    }

    #answer {
        border: solid green;
        padding: 1;
        height: 100%;
        overflow-y: scroll;
    }

    #sources {
        border: solid blue;
        padding: 1;
        margin-top: 1;
        height: 10;
        overflow-y: scroll;
    }

    Button {
        margin: 1;
    }

    #stats {
        layout: horizontal;
        height: 3;
        background: $panel;
        padding: 1;
    }

    .stat {
        width: 1fr;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "clear", "Clear"),
    ]

    def __init__(self):
        super().__init__()
        self.query_count = 0

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        # Title
        yield Static("🔧 Maintenance Documentation Q&A System", id="title")

        # Query Panel
        with Container(id="query-panel", classes="panel"):
            yield Static("Ask a Question:", classes="label")
            yield Input(
                placeholder="Example: How do I replace the air filter?",
                id="question"
            )

            yield Static("Device Type:", classes="label")
            yield Select(
                options=[
                    ("All Devices", "all"),
                    ("Hydraulic Pump", "hydraulic_pump"),
                    ("Air Compressor", "air_compressor"),
                    ("Generator", "generator"),
                    ("CNC Machine", "cnc_machine"),
                ],
                id="device_type"
            )

            with Horizontal():
                yield Button("🔍 Search", id="search_btn", variant="primary")
                yield Button("🗑️  Clear", id="clear_btn")

        # Stats
        with Container(id="stats"):
            yield Static(f"Queries: {self.query_count}", id="stat-queries", classes="stat")
            yield Static("Documents: 127", id="stat-docs", classes="stat")
            yield Static("Status: ✓ Ready", id="stat-status", classes="stat")

        # Results Panel
        with Container(id="result-panel", classes="panel"):
            yield Static("Answer", classes="label")
            yield Static("Ask a question to see the answer here", id="answer")

            yield Static("Sources", classes="label")
            yield Static("Source citations will appear here", id="sources")

        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "search_btn":
            await self.perform_search()
        elif event.button.id == "clear_btn":
            self.action_clear()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input"""
        if event.input.id == "question":
            await self.perform_search()

    async def perform_search(self) -> None:
        """Perform RAG search"""
        question_widget = self.query_one("#question", Input)
        device_widget = self.query_one("#device_type", Select)
        answer_widget = self.query_one("#answer", Static)
        sources_widget = self.query_one("#sources", Static)

        question = question_widget.value.strip()
        device_type = device_widget.value if device_widget.value else "all"

        if not question:
            answer_widget.update("❌ Please enter a question")
            return

        # Show loading
        answer_widget.update("⏳ Searching documentation...")
        sources_widget.update("⏳ Loading sources...")

        # Simulate RAG query
        await self.simulate_rag_query(question, device_type)

    async def simulate_rag_query(self, question: str, device_type: str) -> None:
        """Simulate RAG query with mock data"""
        import asyncio

        # Simulate processing delay
        await asyncio.sleep(0.5)

        self.query_count += 1

        device_name = device_type.replace("_", " ").title() if device_type != "all" else "the equipment"

        # Mock answer
        answer = f"""✅ MAINTENANCE PROCEDURE FOR {device_name.upper()}:

1. SAFETY LOCKOUT
   • Power down equipment completely
   • Apply LOTO procedures
   • Verify zero energy state

2. ACCESS & PREPARATION
   • Tools required: 10mm socket, torque wrench
   • Remove access panels
   • Prepare cleaning materials

3. INSPECTION
   • Visual check for wear/leaks/corrosion
   • Document any anomalies
   • Check fluid levels

4. MAINTENANCE
   • Clean components
   • Apply lubricant (ISO VG 68)
   • Replace worn gaskets

5. TESTING
   • Reassemble with proper torque
   • Remove LOTO
   • Operational test

⚠️  SAFETY: Always follow LOTO procedures
⏱️  Duration: 30-45 minutes
👤 Skill: Qualified technician required"""

        # Mock sources
        sources = f"""📚 SOURCE DOCUMENTS:

1. {device_type.upper()}_Maintenance_Manual_v3.2.pdf
   Pages: 23-25 | Section: Ch. 4 Routine Maintenance
   Relevance: 89% | ⭐⭐⭐⭐⭐

2. Safety_Guidelines_2025.pdf
   Pages: 12-14 | Section: LOTO Procedures
   Relevance: 82% | ⭐⭐⭐⭐

3. {device_type.upper()}_Technical_Specs.pdf
   Pages: 34-35 | Section: Torque Specifications
   Relevance: 76% | ⭐⭐⭐⭐

Query: "{question}"
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Confidence: 85%"""

        # Update UI
        answer_widget = self.query_one("#answer", Static)
        sources_widget = self.query_one("#sources", Static)
        stat_widget = self.query_one("#stat-queries", Static)

        answer_widget.update(answer)
        sources_widget.update(sources)
        stat_widget.update(f"Queries: {self.query_count}")

    def action_clear(self) -> None:
        """Clear all fields"""
        question_widget = self.query_one("#question", Input)
        answer_widget = self.query_one("#answer", Static)
        sources_widget = self.query_one("#sources", Static)

        question_widget.value = ""
        answer_widget.update("Ask a question to see the answer here")
        sources_widget.update("Source citations will appear here")

    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Terminal RAG Interface")
    print("="*60)
    print("\n⌨️  Controls:")
    print("  • Tab: Switch fields")
    print("  • Enter: Submit query")
    print("  • Ctrl+Q: Quit")
    print("  • Ctrl+C: Clear")
    print("\n💡 Tip: This runs entirely in your terminal!\n")

    app = RAGApp()
    app.run()
