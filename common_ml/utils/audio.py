import subprocess
import os
from dataclasses import dataclass
from typing import List

@dataclass
class AudioPart:
    path: str
    duration: float
    start_offset: float

    @property
    def end_offset(self) -> float:
        return self.start_offset + self.duration

class AudioStitcher:
    def __init__(self):
        self.parts: List[AudioPart] = []
        self.total_duration: float = 0.0

    def probe(self, files: List[str], expect_same_length: bool = True):
        """
        Determines the length of each file and stores part info
        """
        self.parts = []
        current_offset = 0.0
        first_part_duration = None

        for i, file_path in enumerate(sorted(files)):
            duration = self._get_duration(file_path)

            if first_part_duration is None:
                first_part_duration = duration

            if expect_same_length and duration != first_part_duration:
                # Allow the last file to be shorter
                if i != len(files) - 1:
                    raise Exception(f"Duration for part {i} is {duration} (expected {first_part_duration})")

            # Create the dataclass instance
            part = AudioPart(
                path=file_path,
                duration=duration,
                start_offset=current_offset
            )
            self.parts.append(part)
            
            if expect_same_length and i != len(files) -1:
                current_offset = len(self.parts) * first_part_duration
            else:
                current_offset += duration

        self.total_duration = current_offset

    def _get_duration(self, file_path: str) -> float:
        """Helper to call ffprobe and extract duration."""
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def stitch(self, start_time: float, end_time: float) -> bytes:
        """
        Returns a byte stream of the stitched audio using the concat demuxer.
        """
        if start_time < 0 or end_time > self.total_duration or start_time >= end_time:
            raise ValueError("Invalid time range requested.")

        # Filter parts using dataclass attributes
        needed_files = [
            f for f in self.parts
            if f.start_offset < end_time and f.end_offset > start_time
        ]

        if not needed_files:
            return b""

        # Create instructions for the concat demuxer
        concat_content = "".join([f"file 'file:{os.path.abspath(f.path)}'\n" for f in needed_files])

        # Calculate seeking logic
        first_file_offset = needed_files[0].start_offset
        relative_start = start_time - first_file_offset
        duration_to_cut = end_time - start_time

        # FFmpeg:
        #  Use concat demuxer to treat files as one stream
        #  -ss for start time, -t for duration
        #  Output to pipe:1 (stdout)
        #  set whitelist for file & pipe
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-protocol_whitelist', 'pipe,file', '-i', 'pipe:0',
            '-ss', str(relative_start), '-t', str(duration_to_cut),
            '-vn', '-acodec', 'copy',
            '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov',
            'pipe:1'
        ]

        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate(input=concat_content.encode())

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

        return stdout
