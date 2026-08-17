import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";

import { useMemo } from "react";
import { architectureData } from "../data/architecture";
import ArchitectureNode from "./ArchitectureNode";
import { ArchitectureItem } from "../types/architecture";

interface Props {
  search: string;
  selectedId: string | null;
  onSelect: (item: ArchitectureItem) => void;
}

const nodeTypes = {
  architecture: ArchitectureNode,
};

export default function ArchitectureCanvas({
  search,
  selectedId,
  onSelect,
}: Props) {
  const initialNodes = useMemo(() => {
    return architectureData.items.map((item, index) => ({
      id: item.id,
      type: "architecture",
      position: {
        x: (index % 4) * 310,
        y: Math.floor(index / 4) * 150,
      },
      data: item,
      selected: item.id === selectedId,
    }));
  }, [selectedId]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);

  const [edges, setEdges, onEdgesChange] = useEdgesState(
    architectureData.edges.map((edge) => ({
      ...edge,
      animated: true,
      style: {
        stroke: "#64748b",
        strokeWidth: 1.5,
      },
    }))
  );

  const filteredNodes = nodes.map((node) => {
    const item = node.data as ArchitectureItem;

    const matches =
      !search ||
      item.name.toLowerCase().includes(search.toLowerCase()) ||
      item.layer.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase());

    return {
      ...node,
      hidden: !matches,
    };
  });

  return (
    <div className="canvas-wrapper">
      <ReactFlow
        nodes={filteredNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => {
          onSelect(node.data as ArchitectureItem);
        }}
        fitView
        attributionPosition="bottom-left"
      >
        <Background gap={24} size={1} color="#1e293b" />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}