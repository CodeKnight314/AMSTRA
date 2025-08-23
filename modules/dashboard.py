from rich.live import Live
from rich.table import Table
from rich.console import Group
from rich.panel import Panel
import threading, time
from rich.align import Align


class Dashboard:
    def __init__(self, host: str, port: int, refresh_per_second: int = 4):
        self.refresh_rate = refresh_per_second
        self.stream_data = {}
        self.lock = threading.Lock()
        self.running = False
        self.address = f"{host}:{port}"
        self.description = f"Simulated Ground station for handling data from Mavic 2 Pro drone.\nServer is running on {self.address}.\nData is sent from drone to ground station via UDP protocol.\nData is processed and displayed on the dashboard."

    def update_stream(self, address, **stats):
        with self.lock:
            if address not in self.stream_data.keys():
                self.stream_data[address] = {}
            self.stream_data[address].update(stats)

    def remove_stream(self, addr):
        with self.lock:
            self.stream_data.pop(addr, None)

    def set_description(self, text: str):
        """Set/update the footer description shown under the table"""
        with self.lock:
            self.description = text

    def _make_table(self):
        table = Table(show_header=True, expand=True)
        table.add_column("Stream", style="cyan")
        table.add_column("Time Created", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Frames", style="green")
        table.add_column("Queue", style="green")
        table.add_column("CPU", style="white")

        with self.lock:
            for addr, stats in self.stream_data.items():
                table.add_row(
                    str(addr),
                    str(stats.get("time_created", "Unknown")),
                    stats.get("status", "Unknown"),
                    str(stats.get("frames", 0)),
                    str(stats.get("queue", 0)),
                    f"{stats.get('cpu', 0)}%",
                )
            
            content = Group(
                table,
                "\n",
                Panel(
                    self.description,
                    border_style="dim",
                    title="Info",
                    title_align="left",
                    expand=True,
                )
            )
            
            return Panel(
                content,
                title="Ground Station Dashboard",
                border_style="blue",
                expand=True,
            )

    def run(self):
        self.running = True
        with Live(self._make_table(), refresh_per_second=self.refresh_rate) as live:
            while self.running:
                live.update(self._make_table())
                time.sleep(1)

    def stop(self):
        self.running = False
