document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // 스크롤을 항상 최하단으로
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        appendMessage('user', message);
        userInput.value = ''; // 입력창 비우기
        sendButton.disabled = true; // 메시지 보내는 동안 버튼 비활성화
        userInput.disabled = true; // 입력창 비활성화

        appendMessage('bot', '답변 생성 중...'); // 로딩 메시지 표시

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // "답변 생성 중..." 메시지를 찾아서 업데이트
            const loadingMessage = chatBox.querySelector('.message.bot:last-child');
            if (loadingMessage && loadingMessage.textContent === '답변 생성 중...') {
                loadingMessage.textContent = data.response;
            } else {
                // 혹시 못 찾으면 새로 추가
                appendMessage('bot', data.response);
            }

        } catch (error) {
            console.error('Error:', error);
            const loadingMessage = chatBox.querySelector('.message.bot:last-child');
            if (loadingMessage && loadingMessage.textContent === '답변 생성 중...') {
                loadingMessage.textContent = "죄송합니다. 챗봇과 통신 중 오류가 발생했습니다.";
            } else {
                appendMessage('bot', "죄송합니다. 챗봇과 통신 중 오류가 발생했습니다.");
            }
        } finally {
            sendButton.disabled = false; // 버튼 다시 활성화
            userInput.disabled = false; // 입력창 다시 활성화
            userInput.focus(); // 입력창에 포커스
            chatBox.scrollTop = chatBox.scrollHeight; // 스크롤을 항상 최하단으로
        }
    }

    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});