import React from "react";
import { DragDropContext, Draggable, Droppable } from "react-beautiful-dnd";
import { AIBlock } from "../models/block";
import AIBlockItem from "./aiblockitem";

type Props = {
  droppableId: string;
  blocks: AIBlock[];
  setBlocks: React.Dispatch<React.SetStateAction<AIBlock[]>>;
};

const Column: React.FC<Props> = ({ droppableId, blocks, setBlocks }) => (
  <Droppable droppableId={droppableId}>
    {(droppableProvided, droppableSnapshot) => (
      <div
        className="bg-gray-400 px-5 py-3 rounded-md"
        ref={droppableProvided.innerRef}
        {...droppableProvided.droppableProps}
      >
        <span className="text-white text-2xl font-semibold">Backlog</span>
        {blocks.map((block, index) => (
          <AIBlockItem
            index={index}
            key={block.id}
            block={block}
            blocks={blocks}
            setBlocks={setBlocks}
          />
        ))}
      </div>
    )}
  </Droppable>
);

export default Column;
