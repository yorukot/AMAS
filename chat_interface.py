import ollama
import json
import time
import asyncio
from datetime import datetime

class ChatInterface:
    def __init__(self):
        self.model = "llama3"
        self.chat_history = []
        self.system_prompt = """你是一個專為視障者設計的AI導航助手。你必須用中文回應所有問題。

當收到環境感知資訊時，你必須根據提供的確切數據來提供導航指引。請嚴格使用提供的距離數值，不要編造或更改任何數據。

你的回應必須按照以下格式提供導航指引：
1. 首先提示最危險或需要立即注意的事項（距離最近的危險）
2. 然後描述周圍環境的主要障礙物，包括確切的距離和方位
3. 最後提供清晰的方向指引，建議最安全的行進方向

你的回應應該簡短、清晰，並遵循以下結構：
「注意！{最危險物體的方位}{確切距離}米處有{物體類型}。{其他物體的方位}{確切距離}米處有{物體類型}。{可通行方位}可通行，建議向{建議方向}行進。」

例如，根據環境資訊：
{
  "objects": [
    {"type": "行人", "position": "前方", "distance": 5},
    {"type": "汽車", "position": "左方", "distance": 3}
  ],
  "traffic_lights": {"status": "紅燈", "position": "前方", "distance": 10},
  "distances": {"walkable_area": "右方", "nearest_obstacle": 2}
}

你應該回答：
「注意！左方3米處有汽車，這是最近的危險。前方5米處有行人，請減速。前方10米處為紅燈。右方道路暢通，建議向右轉。」

請記住，你的回應對視障者的安全至關重要，必須準確、清晰、實用，並且嚴格基於提供的數據。
"""

    async def chat(self, user_message):
        """與用戶進行對話"""
        try:
            # 添加用戶訊息到歷史記錄
            self.chat_history.append({"role": "user", "content": user_message})
            
            # 準備完整的對話歷史
            messages = [{"role": "system", "content": self.system_prompt}] + self.chat_history
            
            # 調用Ollama API
            client = ollama.Client()
            response = client.chat(
                model=self.model,
                messages=messages
            )
            
            # 獲取回應內容
            ai_message = response.message.content
            
            # 添加AI回應到歷史記錄
            self.chat_history.append({"role": "assistant", "content": ai_message})
            
            return ai_message
            
        except Exception as e:
            print(f"對話發生錯誤: {str(e)}")
            return f"抱歉，系統出現了問題: {str(e)}"
    
    def simulate_navigation(self):
        """模擬導航場景"""
        scene = {
            "objects": [
                {"type": "行人", "position": "前方", "distance": 5},
                {"type": "汽車", "position": "左方", "distance": 3}
            ],
            "traffic_lights": {
                "status": "紅燈",
                "position": "前方",
                "distance": 10
            },
            "distances": {
                "walkable_area": "右方",
                "nearest_obstacle": 2
            }
        }
        
        return f"【環境感知資訊】\n{json.dumps(scene, ensure_ascii=False, indent=2)}"

async def main():
    chat_interface = ChatInterface()
    
    print("=== 視障者導航助手 ===")
    print("(輸入 'exit' 結束對話，輸入 'nav' 模擬導航場景，輸入 'help' 查看功能說明)")
    
    while True:
        user_input = input("\n您: ")
        
        if user_input.lower() == 'exit':
            print("謝謝使用，再見！")
            break
        
        if user_input.lower() == 'nav':
            nav_scene = chat_interface.simulate_navigation()
            print(f"\n系統: {nav_scene}")
            user_input = "請根據上述環境資訊，提供導航指引"
            
        if user_input.lower() == 'help':
            print("\n系統: 視障者導航助手功能說明")
            print("1. 即時分析環境感知資訊")
            print("2. 提供清晰、簡潔的語音導引")
            print("3. 優先提示危險或需要立即注意的事項")
            print("4. 協助視障者安全導航")
            print("\n您可以輸入 'nav' 來模擬導航場景，系統將會提供導航指引")
            continue
        
        print("\n正在思考...", end="", flush=True)
        start_time = time.time()
        
        response = await chat_interface.chat(user_input)
        
        # 計算回應時間
        elapsed = time.time() - start_time
        
        # 清除"正在思考..."
        print("\r" + " " * 20 + "\r", end="")
        
        # 顯示回應
        print(f"助手 ({elapsed:.2f}秒): {response}")

if __name__ == "__main__":
    asyncio.run(main()) 