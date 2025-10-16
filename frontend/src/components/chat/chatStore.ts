import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: number;
  file_path?: string;
}

export interface ChatState {
  // Current chat
  currentChatId: string;
  messages: ChatMessage[];
  isLoading: boolean;
  
  // Chat history
  chats: Record<string, ChatMessage[]>;
  
  // Actions
  setCurrentChat: (chatId: string) => void;
  addMessage: (message: ChatMessage) => void;
  setMessages: (messages: ChatMessage[]) => void;
  setLoading: (loading: boolean) => void;
  clearCurrentChat: () => void;
  createNewChat: () => string;
  
  // Local storage persistence
  saveChatToStorage: (chatId: string, messages: ChatMessage[]) => void;
  loadChatFromStorage: (chatId: string) => ChatMessage[];
}

// Generate unique chat ID
const generateChatId = (): string => {
  return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // Initial state
      currentChatId: '',
      messages: [],
      isLoading: false,
      chats: {},
      
      // Actions
      setCurrentChat: (chatId: string) => {
        const state = get();
        
        // Save current chat before switching
        if (state.currentChatId && state.messages.length > 0) {
          state.saveChatToStorage(state.currentChatId, state.messages);
        }
        
        // Load new chat
        const chatMessages = state.loadChatFromStorage(chatId);
        
        set({
          currentChatId: chatId,
          messages: chatMessages,
        });
      },
      
      addMessage: (message: ChatMessage) => {
        set((state) => {
          const newMessages = [...state.messages, message];
          
          // Auto-save to storage
          state.saveChatToStorage(state.currentChatId, newMessages);
          
          return { messages: newMessages };
        });
      },
      
      setMessages: (messages: ChatMessage[]) => {
        set({ messages });
      },
      
      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },
      
      clearCurrentChat: () => {
        set({ messages: [] });
      },
      
      createNewChat: () => {
        const chatId = generateChatId();
        set({
          currentChatId: chatId,
          messages: [],
          isLoading: false,
        });
        return chatId;
      },
      
      // Local storage methods
      saveChatToStorage: (chatId: string, messages: ChatMessage[]) => {
        set((state) => ({
          chats: {
            ...state.chats,
            [chatId]: messages,
          },
        }));
      },
      
      loadChatFromStorage: (chatId: string) => {
        const state = get();
        return state.chats[chatId] || [];
      },
    }),
    {
      name: 'sva-chat-storage', // localStorage key
      partialize: (state) => ({
        chats: state.chats, // Only persist chat history
      }),
    }
  )
);