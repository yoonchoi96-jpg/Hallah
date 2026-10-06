from core.candidates.models import Candidate
from core.music_context.models import SongContext

def test_song_context_versioning():
    context = SongContext(title="Untitled")
    next_context = context.next_version()
    assert context.version == 0
    assert next_context.version == 1
    assert context.title == next_context.title

def test_candidate_is_bound_to_context_version():
    candidate = Candidate(id="candidate-a", direction="identity", intent="Preserve the drum groove.", rationale="Protect rhythmic authority.", parent_context_version=3)
    assert candidate.parent_context_version == 3
