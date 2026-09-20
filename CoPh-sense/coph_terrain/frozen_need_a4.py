"""Deployment adapter for the frozen NEED-conditioned A4 value model."""
from pathlib import Path
import numpy as np
import torch
from .lookahead_bidirectional import NeedPacket, compatible_need
from .lookahead_need_learning import ValueMLP

DEFAULT_A4_CHECKPOINT = (Path(__file__).resolve().parent / "results" /
                         "lookahead_bidirectional_a4" / "history_need_a4_v1.pt")

def legal_need_features(env, sender, packet_id, need, need_age=0):
    """Frozen 19-D map; uses sender observation, owned innovation, delivered NEED."""
    if not isinstance(need, NeedPacket):
        raise TypeError("need must be a decoded NeedPacket")
    packet = env.innovation_packets[packet_id]
    if packet.source != sender or packet_id not in env.owned[sender]:
        raise ValueError("innovation is not locally owned by sender")
    local = env.observations()[sender]
    kin = np.asarray(local["kinematics"], dtype=np.float32)
    sender_features = np.asarray([
        packet.posterior_mean, np.sqrt(packet.posterior_variance),
        packet.posterior_mean-packet.prior_mean,
        float(env.step_index-packet.acquired_step), float(kin[0]), float(kin[1]),
        float(kin[4]), float(kin[5]), float(local["lookahead"]["delta_tau"]),
        float(packet_id in local["acknowledged_evidence_ids"]),
        float(sender == "carrier")], np.float32)
    need_features = np.asarray([
        float(need.region), float(need.modality == "traction"), need.uncertainty,
        need.task_relevance,
        need.latest_useful_time-env.step_index*env.config.dt,
        need.receiver_progress, float(need_age),
        float(packet.region == need.region and packet.modality == need.modality)],
        np.float32)
    return np.concatenate((sender_features, need_features))

class FrozenNeedA4:
    def __init__(self, checkpoint=DEFAULT_A4_CHECKPOINT, device="cpu"):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"frozen A4 checkpoint missing: {self.checkpoint}")
        payload = torch.load(self.checkpoint, map_location=device, weights_only=False)
        self.mean = np.asarray(payload["mean"], np.float32)
        self.scale = np.asarray(payload["scale"], np.float32)
        if self.mean.shape != (19,) or self.scale.shape != (19,):
            raise ValueError("frozen A4 checkpoint has incompatible features")
        self.model = ValueMLP(19).to(device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval(); self.device = device

    @torch.inference_mode()
    def value(self, env, sender, packet_id, need, need_age=0):
        x = legal_need_features(env, sender, packet_id, need, need_age)
        x = torch.as_tensor((x-self.mean)/self.scale, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        return float(self.model(x).item())

    def decide(self, env, sender, packet_id, need, need_age=0):
        value = self.value(env, sender, packet_id, need, need_age)
        causal = compatible_need(env.innovation_packets[packet_id], need,
                                 env.step_index*env.config.dt)
        send = bool(causal and value > 0.)
        return {"action": "send" if send else "hold", "send": send,
                "predicted_send_value": value, "compatible": bool(causal)}
