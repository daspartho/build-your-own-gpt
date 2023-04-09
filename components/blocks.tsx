import React, { useState } from "react";
import { DragDropContext, Draggable, Droppable } from "react-beautiful-dnd";
import { Status, AIBlock, ComponentStatus } from "../models/todo";
import AIBlockItem from "./aiblockitem";

interface Props {
  availableBlocks: AIBlock[];
  setAvailableBlocks: React.Dispatch<React.SetStateAction<AIBlock[]>>;
  userBlocks: AIBlock[];
  setUserBlocks: React.Dispatch<React.SetStateAction<AIBlock[]>>;
}

const Blocks: React.FC<Props> = ({
  availableBlocks,
  setAvailableBlocks,
  userBlocks,
  setUserBlocks,
}) => {
  const [nEmbed, setNEmbed] = useState(128);
  const [blockSize, setBlockSize] = useState(64);
  const [act, setAct] = useState("relu");
  return (
    <div className="flex w-full mt-4 h-[88vh] space-x-4">
      <Droppable droppableId={ComponentStatus.Panel}>
        {(droppableProvided, droppableSnapshot) => (
          <div
            className="bg-black px-5 py-3 rounded-md w-[50%] max-w-[50%] border"
            ref={droppableProvided.innerRef}
            {...droppableProvided.droppableProps}
          >
            <span className="text-white text-3xl font-semibold p-[40px]">
              AI Blocks
            </span>
            {availableBlocks?.map((block, index) => (
              <AIBlockItem
                hasDoneIcon={false}
                index={index}
                key={block?.id}
                block={block}
                blocks={availableBlocks}
                setBlocks={setAvailableBlocks}
              />
            ))}
            {droppableProvided.placeholder}
          </div>
        )}
      </Droppable>
      <Droppable droppableId={ComponentStatus.Model}>
        {(droppableProvided, droppableSnapshot) => (
          <form
            className={`bg-black border px-5 py-3 rounded-md  w-[100%]`}
            onSubmit={(event) => {
              event.preventDefault();
              const reqParams = [nEmbed, blockSize, act];
              const req = {
                userBlocks,
                reqParams,
              };
              console.log(req);
            }}
          >
            <div className="relative static text-black border bg-red-300 flex space-x-8 w-full h-[67.5px] justify-center items-center rounded-md">
              <label htmlFor="nembed">
                n_embed
                <input
                  id="nembed"
                  className="text-white w-[100px] text-center ml-2"
                  type="text"
                  placeholder="0"
                  value={nEmbed}
                  onChange={(e) => setNEmbed(parseInt(e.target.value))}
                />
              </label>
              <label htmlFor="blocksize">
                block_size
                <input
                  id="blocksize"
                  className="text-white w-[100px] text-center ml-2"
                  type="text"
                  placeholder="0"
                  value={blockSize}
                  onChange={(e) => setBlockSize(parseInt(e.target.value))}
                />
              </label>
              <label htmlFor="act">
                act
                <input
                  id="act"
                  className="text-white w-[100px] text-center ml-2"
                  type="text"
                  placeholder=""
                  value={act}
                  onChange={(e) => setAct(e.target.value)}
                />
              </label>
              <button
                type="submit"
                className="absolute right-8 border border-black p-2 rounded-md bg-neutral-800 text-white"
              >
                Train Model
              </button>
            </div>

            {/* <span className="text-white text-2xl font-semibold p-[40px]">
              My Model
            </span> */}
            <div
              className={`mt-[8px] bg-black h-[88%] border px-5 py-3 rounded-md  w-[100%] ${
                droppableSnapshot.isDraggingOver ? "opacity-80" : ""
              }`}
              ref={droppableProvided.innerRef}
              {...droppableProvided.droppableProps}
            >
              {userBlocks?.map((block, index) => (
                <AIBlockItem
                  hasDoneIcon={false}
                  index={index}
                  key={block?.id}
                  block={block}
                  blocks={userBlocks}
                  setBlocks={setUserBlocks}
                />
              ))}
              {droppableProvided.placeholder}
            </div>
          </form>
        )}
      </Droppable>
    </div>
  );
};

export default Blocks;
