import express from "express";
import cors from "cors";


const app = express();
app.use(cors());
app.use(express.json());


app.post("/chat", async (req, res) => {
const { message } = req.body;


// 這裡之後可改成：Ollama / OpenAI / RAG
const reply = `你剛剛說的是：${message}`;


res.json({ reply });
});


app.listen(3001, () => {
console.log("Server running at http://localhost:3001");
});
