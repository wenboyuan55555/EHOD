"""Registry of the event mechanisms evaluated on a frozen HCF state."""
from baselines.event._shared.features import EVENT_FEATURE_DIM
from baselines.event._shared.structured import StructuredEventAdapter
from baselines.event.concat.fusion import ConcatEventAdapter
from baselines.event.event_matrix.fusion import MatrixEventAdapter
from baselines.event.egaf.fusion import LegacyAttentionEventAdapter
from baselines.event.qa_stgacn.fusion import QAQueryEventAdapter
from baselines.event.cdgon.fusion import CDGONEventAdapter
from baselines.event.t3.fusion import T3DualPredictorEventAdapter
from baselines.event.fuse.fusion import FuseCrossAttentionEventAdapter
from baselines.event.ss_moe.fusion import SemanticSpatialMoEEventAdapter
