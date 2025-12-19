const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const messages = document.getElementById("messages");


sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", e => {
if (e.key === "Enter") sendMessage();
});


function addMessage(text, role) {
const div = document.createElement("div");
div.className = `msg ${role}`;
div.textContent = text;
messages.appendChild(div);
messages.scrollTop = messages.scrollHeight;
}


async function sendMessage() {
const text = input.value.trim();
if (!text) return;


addMessage(text, "user");
input.value = "";


const res = await fetch("http://localhost:3001/chat", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ message: text })
});


const data = await res.json();
addMessage(data.reply, "bot");
}