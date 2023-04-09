// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from "next";

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
  if (req.method !== "POST") {
    res.status(405).send({ message: "Only POST requests allowed" });
    return;
  } else {
    try {
      const { model } = JSON.parse(req.body);
      fetch("http://localhost:3001", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(model),
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
