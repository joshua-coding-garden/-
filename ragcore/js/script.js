document.addEventListener('DOMContentLoaded', () => {
    const inputField = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatBox = document.getElementById('chat-box');
    const modal = document.getElementById('source-modal');
    const modalBody = document.getElementById('modal-body');
    const closeBtn = document.querySelector('.close-btn');

    // 關閉 Modal 的功能
    closeBtn.onclick = () => modal.classList.add('hidden');
    window.onclick = (e) => { if (e.target == modal) modal.classList.add('hidden'); }

    async function sendMessage() {
        const question = inputField.value.trim();
        if (!question) return;

        appendMessage(question, 'user');
        inputField.value = '';
        inputField.disabled = true;

        const loadingId = appendMessage('🔍 正在檢索醫療文獻並生成回答...', 'bot', true);

        try {
            const response = await fetch('http://127.0.0.1:5000/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();
            removeMessage(loadingId);

            // 🔥 重點：傳入 sources 參數來渲染按鈕
            appendMessage(data.answer, 'bot', false, data.sources);

        } catch (error) {
            removeMessage(loadingId);
            appendMessage("❌ 連線錯誤", 'bot');
            console.error(error);
        } finally {
            inputField.disabled = false;
            inputField.focus();
        }
    }

    // 新增：顯示來源詳情的函式
    window.showSourceDetails = (sources) => {
        modalBody.innerHTML = ''; // 清空舊內容
        
        sources.forEach(src => {
            const item = document.createElement('div');
            item.className = 'source-item';
            item.innerHTML = `
                <div class="score-badge">相似度: ${src.score}</div>
                <p><strong>片段 ${src.id}:</strong> ${src.content}</p>
            `;
            modalBody.appendChild(item);
        });

        modal.classList.remove('hidden'); // 顯示 Modal
    };

    function appendMessage(text, sender, isLoading = false, sources = []) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        
        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        if (isLoading) bubble.classList.add('loading');
        
        // 使用 marked 解析 Markdown 格式
        bubble.innerHTML = isLoading ? text : marked.parse(text);

        // 🔥 如果有來源資料，在泡泡下方加入按鈕區
        if (sources && sources.length > 0) {
            const linksDiv = document.createElement('div');
            linksDiv.className = 'source-links';
            linksDiv.innerHTML = `<span style="font-size:0.8em; color:#888;">參考來源:</span>`;
            
            // 這裡為了方便，我們把資料暫存到按鈕的 onclick 事件中
            // 注意：實際專案可能用更優雅的方式傳遞資料
            const btn = document.createElement('span');
            btn.className = 'source-tag';
            btn.innerText = `📄 查看 ${sources.length} 個相關片段 (相似度詳情)`;
            
            // 綁定點擊事件
            btn.onclick = () => window.showSourceDetails(sources);
            
            linksDiv.appendChild(btn);
            bubble.appendChild(linksDiv);
        }

        msgDiv.appendChild(bubble);
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    }

    function removeMessage(element) {
        if (element) element.remove();
    }

    sendBtn.addEventListener('click', sendMessage);
    inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});