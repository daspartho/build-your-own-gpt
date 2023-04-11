import React, { useRef, useState, useEffect } from "react";
import { Draggable } from "react-beautiful-dnd";
import { AiFillEdit, AiFillDelete } from "react-icons/ai";
import { MdDone } from "react-icons/md";
import { AIBlock, Status } from "../models/todo";

interface Props {
  hasDoneIcon?: boolean;
  index: number;
  block: AIBlock;
  blocks: AIBlock[];
  setBlocks: React.Dispatch<React.SetStateAction<AIBlock[]>>;
}

const AIBlockItem: React.FC<Props> = ({
  hasDoneIcon = true,
  index,
  block,
  blocks,
  setBlocks,
}) => {
  const [edit, setEdit] = useState<boolean>(false);
  const [editValue, setValue] = useState<number | string>(block.value);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, [edit]);

  const handleEdit = () => {
    if (block.status !== Status.Done && !edit) {
      setEdit(true);
    }
  };

  const handleEditNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(parseFloat(e.target.value));
  };

  const handleEditNameSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBlocks(
      blocks.map((item) =>
        item.id === block.id ? { ...item, value: editValue } : item
      )
    );
    setEdit(false);
  };

  const handleDelete = () => {
    setBlocks(blocks.filter((item) => item.id !== block.id));
  };

  const handleDone = () => {
    //setTodos(todos.map((item) => {
    //  if (item.id == todo.id) {
    //    item.isDone = !item.isDone
    //    return item
    //  }
    //  return item
    //}))
    // or
    setBlocks(
      blocks.map((item) =>
        item.id === block.id ? { ...item, isDone: !item.isDone } : item
      )
    );
  };

  return (
    <Draggable draggableId={block.id.toString()} index={index} key={block.id}>
      {(draggableProvided, draggableSnapshot) => (
        <form
          className={`flex flex-col rounded-md ${block.color} w-[420px] mx-auto p-[20px] mt-[15px] hover:shadow-md`}
          onSubmit={handleEditNameSubmit}
          {...draggableProvided.draggableProps}
          {...draggableProvided.dragHandleProps}
          ref={draggableProvided.innerRef}
        >
          <span className="flex-1">{block.name}</span>
          <div className="w-full flex justify-center items-center text-center ">
            {block.hasInput && (
              <>
                {block.label && <span>{block.label}:</span>}
                {block.isDone ? (
                  <s className="flex-1">{block.value}</s>
                ) : edit ? (
                  <input
                    autoFocus
                    className="w-[50%] text-black text-center px-1 py-2 flex-1 outline-none self-end rounded-md "
                    type="text"
                    ref={inputRef}
                    value={editValue}
                    onChange={handleEditNameChange}
                  />
                ) : (
                  <span className="flex-1">{block.value}</span>
                )}
              </>
            )}

            {block.hasInput && (
              <div className="flex gap-1">
                <span
                  className="ml-[10px] text-[25px] cursor-pointer"
                  onClick={handleEdit}
                >
                  <AiFillEdit />
                </span>

                {hasDoneIcon && (
                  <span
                    className="ml-[10px] text-[25px] cursor-pointer"
                    onClick={handleDone}
                  >
                    <MdDone />
                  </span>
                )}
              </div>
            )}

            <span
              className="ml-[10px] text-[25px] cursor-pointer"
              onClick={handleDelete}
            >
              <AiFillDelete />
            </span>
          </div>
        </form>
      )}
    </Draggable>
  );
};

export default AIBlockItem;
