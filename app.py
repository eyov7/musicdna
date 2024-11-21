import lightning as L
from main import MusicDNAApp

class LightningApp(L.LightningWork):
    def __init__(self):
        super().__init__(parallel=False)
        self.app = None

    def run(self):
        self.app = MusicDNAApp()
        demo = self.app.create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)

app = L.LightningApp(LightningApp())
