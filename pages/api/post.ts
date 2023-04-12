import NextCors from "nextjs-cors";
import type { NextApiRequest, NextApiResponse } from "next";
import { cp } from "fs";

type Data = {
  postResponse: Object;
};

type Error = {
  message: string;
};

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<Data | Error>
) {
  // await NextCors(req, res, {
  //   // Options
  //   methods: ["GET", "HEAD", "PUT", "PATCH", "POST", "DELETE"],
  //   origin: "*",
  //   optionsSuccessStatus: 200, // some legacy browsers (IE11, various SmartTVs) choke on 204
  // });
  console.log(req.body);

  if (req.method !== "POST") {
    res.status(405).send({ message: "Only POST requests allowed" });
    return;
  } else {
    try {
      const model = JSON.stringify(req.body);
      console.log(model);
      fetch("http://localhost:3001/block", {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
        },
        body: model,
      }).then((postResponse) => {
        console.log(postResponse);
        res.json({ postResponse });
      });
      // res.status(200).json({ name: "John Doe" });
    } catch (error) {
      console.log(error);
    }
  }
}
