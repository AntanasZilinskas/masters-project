import { useStore } from '../store';

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000; // 3 seconds
  
  constructor(private url: string) {}
  
  connect() {
    if (this.socket) {
      return;
    }
    
    this.socket = new WebSocket(this.url);
    
    this.socket.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        useStore.getState().updateForecastData(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    this.socket.onclose = (event) => {
      console.log('WebSocket closed, code:', event.code);
      this.socket = null;
      
      // Only attempt to reconnect if it wasn't a clean closure
      if (event.code !== 1000 && event.code !== 1001) {
        this.scheduleReconnect();
      }
    };
    
    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
  
  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached, giving up');
      return;
    }
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect();
    }, this.reconnectDelay);
  }
  
  // For development/testing, we can simulate messages
  simulateMessage(data: any) {
    useStore.getState().updateForecastData(data);
  }
}

// Create a singleton instance
export const wsService = new WebSocketService('ws://localhost:8080/ws/forecast');

// Custom hook for connecting/disconnecting in React components
export const useWebSocket = () => {
  return {
    connect: () => wsService.connect(),
    disconnect: () => wsService.disconnect(),
    simulateMessage: (data: any) => wsService.simulateMessage(data)
  };
}; 