import type { NextPage } from "next";
import Head from "next/head";

import { useEffect, useState } from "react";

import { DragDropContext, DropResult } from "react-beautiful-dnd";

import Blocks from "../components/blocks";

import { Status, AIBlock, ComponentStatus } from "../models/todo";

import styles from "../styles/Home.module.css";

const defaultComponents: AIBlock[] = [
  {
    id: 0,
    name: "self attention",
    label: "n_head",
    value: 4,
    status: Status.Backlog,
    color: "bg-blue-600",
    hasInput: true,
    isDone: false,
  },
  {
    id: 1,
    name: "layernorm",
    value: "layernorm",
    status: Status.Backlog,
    color: "bg-red-600",
    hasInput: false,
    isDone: false,
  },
  {
    id: 2,
    name: "dropout",
    label: "dropout",
    value: 0.1,
    status: Status.Backlog,
    color: "bg-amber-600",
    hasInput: true,
    isDone: false,
  },
  {
    id: 3,
    name: "mlp",
    label: "n_proj",
    value: 256,
    status: Status.Backlog,
    color: "bg-purple-600",
    hasInput: true,
    isDone: false,
  },
];

const Home: NextPage = () => {
  const [name, setName] = useState<string>("");
  const [availableBlocks, setAvailableBlocks] =
    useState<AIBlock[]>(defaultComponents);
  const [userBlocks, setUserBlocks] = useState<AIBlock[]>([]);

  useEffect(() => {
    // let availableBlocks = window.localStorage.getItem("availableBlocks");
    // if (availableBlocks) {
    //   let parsed = JSON.parse(availableBlocks);
    //   setBacklogTodos(parsed);
    // }
    let userBlocks = window.localStorage.getItem("userBlocks");
    if (userBlocks) {
      let parsed = JSON.parse(userBlocks);
      setUserBlocks(parsed);
    }
  }, []);

  const addNewTodo = (e: React.FormEvent) => {
    e.preventDefault();
    if (name) {
      const newTodo = {
        id: Date.now(),
        name,
        value: 0,
        status: Status.Backlog,
        color: "blue-600",
        hasInput: true,
        isDone: false,
      };

      setUserBlocks([...userBlocks, newTodo]);

      setName("");
    }
  };

  const onDragEndHandler = (result: DropResult) => {
    const { destination, source } = result;

    if (
      !destination ||
      (destination.droppableId === source.droppableId &&
        destination.index === source.index)
    )
      return;

    let add,
      backlog = availableBlocks,
      active = userBlocks;

    switch (source.droppableId) {
      case ComponentStatus.Panel:
        add = availableBlocks[source.index];
        backlog.splice(source.index, 1);
        break;
      case ComponentStatus.Model:
        add = active[source.index];
        active.splice(source.index, 1);
        break;
    }

    if (add) {
      switch (destination.droppableId) {
        case ComponentStatus.Panel:
          backlog.splice(destination.index, 0, add);
          break;
        case ComponentStatus.Model:
          active.splice(destination.index, 0, add);
          break;
      }
    }

    setAvailableBlocks(backlog);
    setUserBlocks(active);

    if (window) {
      window.localStorage.setItem("availableBlocks", JSON.stringify(backlog));
      window.localStorage.setItem("userBlocks", JSON.stringify(active));
    }
  };

  return (
    <>
      <Head>
        <title>AI Blocks</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <DragDropContext onDragEnd={onDragEndHandler}>
        <div className={styles.container}>
          <div className="flex flex-col justify-center items-center  min-h-screen">
            <Blocks
              availableBlocks={availableBlocks}
              setAvailableBlocks={setAvailableBlocks}
              userBlocks={userBlocks}
              setUserBlocks={setUserBlocks}
            />
          </div>
        </div>
      </DragDropContext>
    </>
  );
};

Home.getInitialProps = async ({ req }) => {
  console.log("req, ");
  return {};
};

export default Home;
