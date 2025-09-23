import os
import re
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from mistralai import Mistral


class OcrParser:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)

    # ---------- Helpers ----------
    @staticmethod
    def convert2png(base64_string: str, image_file_name: str) -> str:
        """Convert a base64 image string to a PNG file."""
        m = re.match(r"^data:(image/[\w.+-]+);base64,", base64_string)
        base64_payload = base64_string.split(",", 1)[1] if m else base64_string
        base64_payload = re.sub(r"\s+", "", base64_payload)
        base64_payload += "=" * (-len(base64_payload) % 4)

        try:
            image_bytes = base64.b64decode(base64_payload)
        except Exception as e:
            raise ValueError(f"Base64 decode failed. {e}")

        try:
            from PIL import Image

            img = Image.open(BytesIO(image_bytes))
            img.save(image_file_name, format="PNG")
        except ImportError:
            with open(image_file_name, "wb") as f:
                f.write(image_bytes)
        return image_file_name

    @staticmethod
    def extract_title_from_markdown(md: str) -> str | None:
        """Extract first-level Markdown heading (# Title)."""
        m = re.search(r"^[ \t]*#[ \t]+(.+?)\s*$", md, flags=re.M)
        if m:
            return m.group(1).strip()
        m = re.search(r"#[ \t]+([\s\S]*?)\n[ \t]*#", md)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

    @staticmethod
    def safe_fs_name(name: str) -> str:
        """Filesystem-safe folder/file name."""
        name = name.strip()
        name = re.sub(r"[^\w\-. ]+", "_", name)
        return name or "document"

    # ---------- Core async processing ----------
    async def process_pdf(self, file_name: str):
        # Step 1: upload PDF
        uploaded = await self.client.files.upload_async(
            file={
                "file_name": f"{file_name}.pdf",
                "content": open(f"{file_name}.pdf", "rb"),
            },
            purpose="ocr",
        )

        # Step 2: signed URL
        file_id = getattr(uploaded, "id", None) or uploaded["id"]
        signed_url = await self.client.files.get_signed_url_async(file_id=file_id)

        # Step 3: OCR
        ocr_response = await self.client.ocr.process_async(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url.url},
            include_image_base64=True,
        )

        # Step 4: save outputs
        title = (
            self.extract_title_from_markdown(ocr_response.pages[0].markdown)
            or "Untitled"
        )
        title = self.safe_fs_name(title)
        Path(title).mkdir(exist_ok=True)

        final_string = ""
        for page in ocr_response.pages:
            final_string += page.markdown
            for image in page.images:
                self.convert2png(image.image_base64, f"{title}/{image.id}.png")

        (Path(title) / "output.md").write_text(final_string, encoding="utf-8")
        print(f"✅ Finished {file_name} → {title}/output.md")

    async def run(self, files: list[str]):
        await asyncio.gather(*(self.process_pdf(f) for f in files))


# In script:
# if __name__ == "__main__":
#     parser = OcrParser()
#     asyncio.run(parser.run(["CiteMe", "LitSearch"]))
