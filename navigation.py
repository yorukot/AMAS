import json
import ollama
import time

class NavigationLLM:
    def __init__(self):
        self.model = "phi"
        self.system_prompt = """你是一個專門協助視障者的AI導航助手。你的主要任務是：
1. 將環境感知資訊轉換為清晰、簡潔的語音導引
2. 優先提示危險或需要立即注意的事項
3. 使用自然且容易理解的語言
4. 保持指示簡短但資訊完整

當收到環境資訊時，請依照以下原則生成回應：
- 優先順序：危險 > 障礙物 > 方向指引 > 環境描述
- 距離表達：使用具體的公尺數
- 方向表達：使用「前方」、「左方」、「右方」等明確方位
- 語氣：保持冷靜專業，但在危險情況要明確強調

請以「{方位}{距離}{物件}{動作建議}」的結構來組織回應。"""

    async def generate_navigation_guidance(self, scene_info):
        """生成導航指引"""
        try:
            # 組合完整的提示詞
            messages = [{
                'role': 'system',
                'content': self.system_prompt
            }, {
                'role': 'user',
                'content': json.dumps(scene_info, ensure_ascii=False)
            }]
            
            # 使用正確的Ollama API
            client = ollama.Client()
            response = client.chat(
                model=self.model,
                messages=messages
            )
            
            # 使用正確的方式獲取回應内容
            return response.message.content
            
        except Exception as e:
            print(f"生成導航指引時發生錯誤: {str(e)}")
            return "系統暫時無法生成導航指引，請稍後再試。"

    def format_scene_info(self, yolo_results, cnn_results, midas_results):
        """格式化場景資訊"""
        return {
            "objects": yolo_results,
            "traffic_lights": cnn_results,
            "distances": midas_results
        }

# 使用範例
async def main():
    # 初始化導航LLM
    nav_llm = NavigationLLM()
    
    # 模擬場景資訊
    sample_scene = {
        "objects": [
            {"type": "person", "position": "前方", "distance": 5},
            {"type": "car", "position": "左方", "distance": 3}
        ],
        "traffic_lights": {
            "status": "red",
            "position": "前方",
            "distance": 10
        },
        "distances": {
            "walkable_area": "right",
            "nearest_obstacle": 2
        }
    }
    
    # 生成導航指引
    guidance = await nav_llm.generate_navigation_guidance(sample_scene)
    print("導航指引:", guidance)

    # 模擬連續場景變化
    time.sleep(1)  # 模擬處理時間間隔
    
    # 場景變化示例
    sample_scene["objects"].append({
        "type": "bicycle",
        "position": "右方",
        "distance": 2
    })
    
    guidance = await nav_llm.generate_navigation_guidance(sample_scene)
    print("更新的導航指引:", guidance)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
