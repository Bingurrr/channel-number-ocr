"""Temporal ROI history for channel-number candidate regions.

This module tracks *where* channel-number-like UI regions appear across frames.
It intentionally does not use annotation JSON or ground-truth boxes. The only
inputs are model-generated observations such as YOLO boxes and OCR candidate
boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]
TIME_RE = re.compile(r"\b\d{1,2}\s*:\s*\d{2}\b|(?:am|pm)", flags=re.I)
SEARCH = "SEARCH"
LOCK_CANDIDATE = "LOCK_CANDIDATE"
LOCKED = "LOCKED"
UNLOCK_COOLDOWN = "UNLOCK_COOLDOWN"


def digits(text: Any) -> str:
    return "".join(ch for ch in str(text) if "0" <= ch <= "9")


def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def overlap_min_fraction(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(bbox_area(a), bbox_area(b))
    return 0.0 if smaller <= 0 else inter / smaller


def center_distance_norm(a: BBox, b: BBox, width: int, height: int) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.sqrt(((acx - bcx) / max(1, width)) ** 2 + ((acy - bcy) / max(1, height)) ** 2)


def smooth_box(old: BBox, new: BBox, alpha: float) -> BBox:
    return tuple((1.0 - alpha) * old[i] + alpha * new[i] for i in range(4))  # type: ignore[return-value]


def is_numeric_like(text: Any) -> bool:
    value = digits(text)
    if not (1 <= len(value) <= 4):
        return False
    return not TIME_RE.search(str(text))


@dataclass
class RoiObservation:
    bbox: BBox
    confidence: float
    source: str
    numeric_like: bool
    yolo_class_id: Optional[int] = None
    candidate_id: Optional[str] = None
    text: str = ""


@dataclass
class RoiTrack:
    track_id: str
    bbox: BBox
    first_frame: int
    last_frame: int
    seen_count: int = 1
    missed_count: int = 0
    numeric_like_count: int = 0
    yolo_support_count: int = 0
    ocr_support_count: int = 0
    channel_detector_count: int = 0
    confidence_sum: float = 0.0
    update_count: int = 0
    jitter_sum: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update(self, obs: RoiObservation, frame_index: int, frame_width: int, frame_height: int, alpha: float) -> None:
        prev = self.bbox
        self.bbox = smooth_box(self.bbox, obs.bbox, alpha)
        self.last_frame = frame_index
        self.seen_count += 1
        self.missed_count = 0
        self.update_count += 1
        self.confidence_sum += obs.confidence
        self.jitter_sum += center_distance_norm(prev, obs.bbox, frame_width, frame_height)
        if obs.numeric_like:
            self.numeric_like_count += 1
        if obs.source.startswith("yolo"):
            self.yolo_support_count += 1
        else:
            self.ocr_support_count += 1
        if obs.yolo_class_id in (0, 3):
            self.channel_detector_count += 1
        self.history.append(
            {
                "frame_index": frame_index,
                "bbox_xyxy": [round(v, 3) for v in obs.bbox],
                "source": obs.source,
                "numeric_like": obs.numeric_like,
                "confidence": round(obs.confidence, 6),
                "candidate_id": obs.candidate_id,
                "text": obs.text,
            }
        )
        self.history = self.history[-20:]

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / max(1, self.update_count)

    @property
    def numeric_ratio(self) -> float:
        return self.numeric_like_count / max(1, self.seen_count)

    @property
    def yolo_ratio(self) -> float:
        return self.yolo_support_count / max(1, self.seen_count)

    @property
    def ocr_ratio(self) -> float:
        return self.ocr_support_count / max(1, self.seen_count)

    @property
    def stability(self) -> float:
        avg_jitter = self.jitter_sum / max(1, self.update_count)
        return max(0.0, min(1.0, 1.0 - avg_jitter / 0.08))

    def score(self, frame_index: int, warmup_frames: int) -> float:
        age = max(1, frame_index - self.first_frame + 1)
        persistence = min(1.0, self.seen_count / max(1, min(age, warmup_frames)))
        recency = max(0.0, 1.0 - self.missed_count / 6.0)
        return (
            0.30 * persistence
            + 0.25 * self.numeric_ratio
            + 0.20 * self.stability
            + 0.15 * min(1.0, self.yolo_ratio + 0.35 * self.channel_detector_count / max(1, self.seen_count))
            + 0.10 * min(1.0, self.ocr_ratio)
        ) * recency

    def to_dict(self, frame_index: int, warmup_frames: int) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox_xyxy": [round(v, 3) for v in self.bbox],
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "seen_count": self.seen_count,
            "missed_count": self.missed_count,
            "numeric_like_count": self.numeric_like_count,
            "yolo_support_count": self.yolo_support_count,
            "ocr_support_count": self.ocr_support_count,
            "channel_detector_count": self.channel_detector_count,
            "avg_confidence": round(self.avg_confidence, 6),
            "numeric_ratio": round(self.numeric_ratio, 6),
            "stability": round(self.stability, 6),
            "history_score": round(self.score(frame_index, warmup_frames), 6),
            "recent_observations": self.history[-5:],
        }


class RoiHistoryTracker:
    def __init__(
        self,
        *,
        match_iou: float = 0.15,
        match_distance: float = 0.075,
        smooth_alpha: float = 0.35,
        max_missed: int = 12,
        min_seen: int = 3,
        stable_score: float = 0.68,
        warmup_frames: int = 8,
    ) -> None:
        self.match_iou = match_iou
        self.match_distance = match_distance
        self.smooth_alpha = smooth_alpha
        self.max_missed = max_missed
        self.min_seen = min_seen
        self.stable_score = stable_score
        self.warmup_frames = warmup_frames
        self.tracks: List[RoiTrack] = []
        self._next_id = 1

    def update(
        self,
        observations: Sequence[RoiObservation],
        *,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> List[RoiTrack]:
        unmatched = set(range(len(self.tracks)))
        assigned_observations = set()

        for obs_index, obs in enumerate(observations):
            best_index = None
            best_score = -1.0
            for track_index in list(unmatched):
                track = self.tracks[track_index]
                ov = iou(obs.bbox, track.bbox)
                dist = center_distance_norm(obs.bbox, track.bbox, frame_width, frame_height)
                if ov < self.match_iou and dist > self.match_distance:
                    continue
                score = ov + max(0.0, self.match_distance - dist)
                if score > best_score:
                    best_score = score
                    best_index = track_index
            if best_index is None:
                continue
            self.tracks[best_index].update(obs, frame_index, frame_width, frame_height, self.smooth_alpha)
            unmatched.remove(best_index)
            assigned_observations.add(obs_index)

        for track_index in unmatched:
            self.tracks[track_index].missed_count += 1

        for obs_index, obs in enumerate(observations):
            if obs_index in assigned_observations:
                continue
            track = RoiTrack(
                track_id=f"roi_{self._next_id:04d}",
                bbox=obs.bbox,
                first_frame=frame_index,
                last_frame=frame_index,
                numeric_like_count=1 if obs.numeric_like else 0,
                yolo_support_count=1 if obs.source.startswith("yolo") else 0,
                ocr_support_count=0 if obs.source.startswith("yolo") else 1,
                channel_detector_count=1 if obs.yolo_class_id in (0, 3) else 0,
                confidence_sum=obs.confidence,
                update_count=1,
                history=[
                    {
                        "frame_index": frame_index,
                        "bbox_xyxy": [round(v, 3) for v in obs.bbox],
                        "source": obs.source,
                        "numeric_like": obs.numeric_like,
                        "confidence": round(obs.confidence, 6),
                        "candidate_id": obs.candidate_id,
                        "text": obs.text,
                    }
                ],
            )
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [track for track in self.tracks if track.missed_count <= self.max_missed]
        self.tracks.sort(key=lambda t: t.score(frame_index, self.warmup_frames), reverse=True)
        return self.tracks

    def stable_tracks(self, frame_index: int) -> List[RoiTrack]:
        return [
            track
            for track in self.tracks
            if track.seen_count >= self.min_seen and track.score(frame_index, self.warmup_frames) >= self.stable_score
        ]

    def snapshot(self, frame_index: int) -> Dict[str, Any]:
        return {
            "tracks": [track.to_dict(frame_index, self.warmup_frames) for track in self.tracks],
            "stable_tracks": [track.track_id for track in self.stable_tracks(frame_index)],
        }


def observations_from_yolo(yolo_boxes: Iterable[Dict[str, Any]]) -> List[RoiObservation]:
    observations: List[RoiObservation] = []
    for box in yolo_boxes:
        class_id = int(box.get("class_id", -1))
        if class_id not in (0, 3):
            continue
        observations.append(
            RoiObservation(
                bbox=tuple(float(v) for v in box["bbox_xyxy"]),
                confidence=float(box.get("conf", 1.0)),
                source=f"yolo_class_{class_id}",
                numeric_like=True,
                yolo_class_id=class_id,
                candidate_id=str(box.get("id", "")),
            )
        )
    return observations


def observations_from_candidates(candidates: Iterable[Dict[str, Any]]) -> List[RoiObservation]:
    observations: List[RoiObservation] = []
    for candidate in candidates:
        if "bbox_xyxy" not in candidate:
            continue
        numeric = is_numeric_like(candidate.get("text", ""))
        if not numeric:
            continue
        observations.append(
            RoiObservation(
                bbox=tuple(float(v) for v in candidate["bbox_xyxy"]),
                confidence=float(candidate.get("ocr_conf", candidate.get("confidence", 0.5))),
                source=str(candidate.get("source", "ocr")),
                numeric_like=True,
                candidate_id=str(candidate.get("id", "")),
                text=str(candidate.get("text", "")),
            )
        )
    return observations


def best_track_for_bbox(
    bbox: BBox,
    stable_tracks: Sequence[RoiTrack],
    *,
    frame_width: int,
    frame_height: int,
    min_overlap: float = 0.25,
    max_distance: float = 0.08,
) -> Tuple[Optional[RoiTrack], float]:
    best: Optional[RoiTrack] = None
    best_score = 0.0
    for track in stable_tracks:
        ov = overlap_min_fraction(bbox, track.bbox)
        dist = center_distance_norm(bbox, track.bbox, frame_width, frame_height)
        if ov < min_overlap and dist > max_distance:
            continue
        proximity = max(0.0, 1.0 - dist / max_distance)
        score = 0.65 * ov + 0.35 * proximity
        if score > best_score:
            best = track
            best_score = score
    return best, best_score


@dataclass
class TemporalSlotState:
    """State machine for bbox-only temporal slot memory.

    This intentionally locks only position. Candidate text is retained only in
    debug fields so a previous channel number value cannot become an inference
    prior.
    """

    state: str = SEARCH
    locked_slot_bbox: Optional[BBox] = None
    locked_slot_confidence: float = 0.0
    locked_slot_age: int = 0
    frames_since_seen: int = 0
    recent_observations: List[Dict[str, Any]] = field(default_factory=list)
    last_transition_reason: str = "initialized"
    recent_candidate_texts: List[str] = field(default_factory=list)
    history_track_id: str = ""
    history_score: float = 0.0

    def is_locked(self) -> bool:
        return self.state == LOCKED and self.locked_slot_bbox is not None

    def _transition(self, next_state: str, reason: str) -> None:
        if next_state != self.state:
            self.last_transition_reason = reason
        else:
            self.last_transition_reason = reason
        self.state = next_state

    def _valid_track_shape(
        self,
        track: RoiTrack,
        *,
        frame_width: int,
        frame_height: int,
        max_bbox_area_ratio: float,
        max_bbox_aspect: float,
    ) -> bool:
        area_ratio = bbox_area(track.bbox) / max(1.0, float(frame_width * frame_height))
        if area_ratio <= 0.0 or area_ratio > max_bbox_area_ratio:
            return False
        w = max(1.0, track.bbox[2] - track.bbox[0])
        h = max(1.0, track.bbox[3] - track.bbox[1])
        aspect = w / h
        if aspect > max_bbox_aspect or aspect < 0.15:
            return False
        return True

    def update_from_tracks(
        self,
        tracks: Sequence[RoiTrack],
        *,
        frame_index: int,
        frame_width: int,
        frame_height: int,
        min_lock_observations: int = 3,
        min_track_score: float = 0.68,
        min_lock_iou: float = 0.4,
        max_missed_before_unlock: int = 5,
        max_bbox_area_ratio: float = 0.16,
        max_bbox_aspect: float = 7.0,
        warmup_frames: int = 8,
    ) -> "TemporalSlotState":
        best: Optional[RoiTrack] = None
        best_score = -1.0
        blocked_shape = False
        for track in tracks:
            score = track.score(frame_index, warmup_frames)
            if not self._valid_track_shape(
                track,
                frame_width=frame_width,
                frame_height=frame_height,
                max_bbox_area_ratio=max_bbox_area_ratio,
                max_bbox_aspect=max_bbox_aspect,
            ):
                blocked_shape = True
                continue
            if track.seen_count < min_lock_observations or score < min_track_score:
                if score > best_score:
                    best = track
                    best_score = score
                continue
            if score > best_score:
                best = track
                best_score = score

        eligible = (
            best is not None
            and best.seen_count >= min_lock_observations
            and best_score >= min_track_score
        )

        if eligible and best is not None:
            if self.locked_slot_bbox is not None:
                lock_iou = iou(best.bbox, self.locked_slot_bbox)
                lock_dist = center_distance_norm(best.bbox, self.locked_slot_bbox, frame_width, frame_height)
                if self.state == LOCKED and lock_iou < min_lock_iou and lock_dist > 0.10:
                    self.frames_since_seen += 1
                    if self.frames_since_seen > max_missed_before_unlock:
                        self.locked_slot_bbox = None
                        self.locked_slot_confidence = 0.0
                        self.locked_slot_age = 0
                        self.history_track_id = ""
                        self.history_score = 0.0
                        self._transition(UNLOCK_COOLDOWN, "locked_track_moved_or_conflicted")
                    else:
                        self._transition(LOCKED, "locked_track_temporarily_conflicted")
                    return self
            self.locked_slot_bbox = best.bbox
            self.locked_slot_confidence = best.avg_confidence
            self.locked_slot_age = self.locked_slot_age + 1 if self.state == LOCKED else 1
            self.frames_since_seen = 0
            self.history_track_id = best.track_id
            self.history_score = max(0.0, min(1.0, best_score))
            self.recent_observations = best.history[-5:]
            self.recent_candidate_texts = [
                str(obs.get("text", ""))
                for obs in self.recent_observations
                if str(obs.get("text", "")).strip()
            ][-5:]
            self._transition(LOCKED, "lock_confirmed")
            return self

        if best is not None and best.seen_count > 1:
            self.recent_observations = best.history[-5:]
            self.recent_candidate_texts = [
                str(obs.get("text", ""))
                for obs in self.recent_observations
                if str(obs.get("text", "")).strip()
            ][-5:]
            self.history_track_id = best.track_id
            self.history_score = max(0.0, min(1.0, best_score))
            self.frames_since_seen = 0
            self._transition(LOCK_CANDIDATE, "track_seen_but_not_lockable")
            return self

        if self.state in (LOCKED, LOCK_CANDIDATE):
            self.frames_since_seen += 1
            if self.frames_since_seen > max_missed_before_unlock:
                self.locked_slot_bbox = None
                self.locked_slot_confidence = 0.0
                self.locked_slot_age = 0
                self.history_track_id = ""
                self.history_score = 0.0
                self._transition(UNLOCK_COOLDOWN, "max_missed_before_unlock")
            else:
                self._transition(self.state, "temporarily_missing")
            return self

        if self.state == UNLOCK_COOLDOWN:
            self.frames_since_seen += 1
            if self.frames_since_seen > 1:
                self.frames_since_seen = 0
                self._transition(SEARCH, "cooldown_complete")
            else:
                self._transition(UNLOCK_COOLDOWN, "cooldown")
            return self

        self._transition(SEARCH, "no_lockable_track_shape_blocked" if blocked_shape else "no_lockable_track")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "locked_slot_bbox": None
            if self.locked_slot_bbox is None
            else [round(v, 3) for v in self.locked_slot_bbox],
            "locked_slot_confidence": round(float(self.locked_slot_confidence), 6),
            "locked_slot_age": self.locked_slot_age,
            "frames_since_seen": self.frames_since_seen,
            "recent_observations": self.recent_observations[-5:],
            "last_transition_reason": self.last_transition_reason,
            "recent_candidate_texts": self.recent_candidate_texts[-5:],
            "history_track_id": self.history_track_id,
            "history_score": round(float(self.history_score), 6),
        }
