import { Store } from '@tauri-apps/plugin-store';
import { useChatStore } from './chat/chatStore';

// Tauri persistent storage for app preferences and extended data
class AppStorage {
  private store: Store | null = null;
  
  async initStore() {
    if (!this.store) {
      this.store = await Store.load('sva-app.dat');
    }
    return this.store;
  }
  
  // User preferences
  async saveUserPreferences(preferences: {
    theme?: 'light' | 'dark' | 'auto';
    defaultAnalysisTypes?: string[];
    autoSaveChats?: boolean;
    maxChatHistory?: number;
  }) {
    const store = await this.initStore();
    await store.set('user_preferences', preferences);
    await store.save();
  }
  
  async getUserPreferences() {
    const store = await this.initStore();
    return await store.get('user_preferences') || {
      theme: 'auto',
      defaultAnalysisTypes: ['transcription', 'vision', 'generation'],
      autoSaveChats: true,
      maxChatHistory: 100,
    };
  }
  
  // File upload history
  async saveFileUploadHistory(file: {
    name: string;
    path: string;
    size: number;
    uploadDate: number;
    analysisResult?: any;
  }) {
    const store = await this.initStore();
    const history = (await store.get('file_upload_history') as any[]) || [];
    history.push(file);
    
    // Keep only last 50 uploads
    if (history.length > 50) {
      history.splice(0, history.length - 50);
    }
    
    await store.set('file_upload_history', history);
    await store.save();
  }
  
  async getFileUploadHistory() {
    const store = await this.initStore();
    return (await store.get('file_upload_history') as any[]) || [];
  }
  
  // Application state
  async saveAppState(state: {
    lastChatId?: string;
    serverConnection?: boolean;
    lastServerCheck?: number;
  }) {
    const store = await this.initStore();
    await store.set('app_state', state);
    await store.save();
  }
  
  async getAppState() {
    const store = await this.initStore();
    return await store.get('app_state') || {
      lastChatId: null,
      serverConnection: false,
      lastServerCheck: 0,
    };
  }
  
  // Chat backup to Tauri storage (in addition to browser localStorage)
  async backupChatHistory() {
    const store = await this.initStore();
    const chatStore = useChatStore.getState();
    const backup = {
      chats: chatStore.chats,
      backupDate: Date.now(),
    };
    
    await store.set('chat_backup', backup);
    await store.save();
    
    console.log('💾 Chat history backed up to Tauri storage');
  }
  
  async restoreChatHistory() {
    const store = await this.initStore();
    const backup = await store.get('chat_backup') as any;
    if (backup && backup.chats) {
      const chatStore = useChatStore.getState();
      
      // Merge with existing chats
      const mergedChats = { ...backup.chats, ...chatStore.chats };
      
      // Update store
      useChatStore.setState({ chats: mergedChats });
      
      console.log('📥 Chat history restored from Tauri storage');
      return true;
    }
    return false;
  }
  
  // Clear all data
  async clearAllData() {
    const store = await this.initStore();
    await store.clear();
    await store.save();
    console.log('🗑️ All app data cleared');
  }
}

// Export singleton instance
export const appStorage = new AppStorage();

// Helper hook for using app storage in React components
export const useAppStorage = () => {
  return {
    saveUserPreferences: appStorage.saveUserPreferences.bind(appStorage),
    getUserPreferences: appStorage.getUserPreferences.bind(appStorage),
    saveFileUploadHistory: appStorage.saveFileUploadHistory.bind(appStorage),
    getFileUploadHistory: appStorage.getFileUploadHistory.bind(appStorage),
    saveAppState: appStorage.saveAppState.bind(appStorage),
    getAppState: appStorage.getAppState.bind(appStorage),
    backupChatHistory: appStorage.backupChatHistory.bind(appStorage),
    restoreChatHistory: appStorage.restoreChatHistory.bind(appStorage),
    clearAllData: appStorage.clearAllData.bind(appStorage),
  };
};