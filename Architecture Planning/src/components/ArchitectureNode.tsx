import { Handle, Position } from "@xyflow/react";
import { ArchitectureItem } from "../types/architecture";

interface Props {
  data: ArchitectureItem;
  selected?: boolean;
}

export default function ArchitectureNode({ data, selected }: Props) {
  const statusClass =
    data.status === "MVP"
      ? "mvp"
      : data.status === "Future"
      ? "future"
      : "both";

  return (
    <div className={`architecture-node ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Left} />

      <div className="node-layer">{data.layer}</div>

      <div className="node-title">{data.name}</div>

      <div className="node-type">{data.type}</div>

      {data.trigger && (
        <div className="node-trigger">
          {data.trigger}
        </div>
      )}

      {data.status && (
        <div className={`node-status ${statusClass}`}>
          {data.status}
        </div>
      )}

      <Handle type="source" position={Position.Right} />
    </div>
  );
}