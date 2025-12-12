/**
 * API Service for PropGPT Frontend
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

/**
 * Get available items for a comparison type
 */
// export const getComparisonItems = async (comparisonType, search = '', limit = 100) => {
//     const response = await api.get('/comparison-items/', {
//         params: {
//             comparison_type: comparisonType,
//             search: search,
//             limit: limit
//         }
     
//     });
//     return response.data;
// };

export const getComparisonItems = async (comparisonType, search = '', limit = 100) => {
  try {
    console.log("Calling getComparisonItems with:", {
      comparisonType,
      search,
      limit,
    });

    const response = await api.get("/comparison-items/", {
      params: {
        comparison_type: comparisonType,
        search: search,
        limit: limit,
      },
    });

    console.log("Response from /comparison-items/:", response.data);
    return response.data;

  } catch (error) {
    console.error("Error in getComparisonItems:", error);
    throw error;
  }
};


















/**
 * Submit a query for processing
 */
export const submitQuery = async (queryData) => {
    const response = await api.post('/query/', queryData);
    return response.data;
};

/**
 * Submit feedback (thumbs up/down)
 */
export const submitFeedback = async (feedbackData) => {
    const response = await api.post('/feedback/', feedbackData);
    return response.data;
};

/**
 * Get cache statistics
 */
export const getCacheStats = async () => {
    const response = await api.get('/cache/stats/');
    return response.data;
};

/**
 * Clear cache
 */
export const clearCache = async () => {
    const response = await api.delete('/cache/clear/');
    return response.data;
};

/**
 * Submit a query with streaming response
 * @param {Object} queryData - The query payload
 * @param {Function} onChunk - Callback for token chunks (text)
 * @param {Function} onStatus - Callback for status updates (text)
 * @param {Function} onMetadata - Callback for final metadata (object)
 * @param {Function} onError - Callback for errors
 * @returns {Promise<void>}
 */
export const streamQuery = async (queryData, onChunk, onStatus, onMetadata, onError) => {
    try {
        const response = await fetch(`${API_BASE_URL}/query/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ...queryData, stream: true }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMsg = response.statusText;
            try {
                const errorJson = JSON.parse(errorText);
                errorMsg = errorJson.error || errorJson.detail || errorMsg;
            } catch (e) { }
            throw new Error(errorMsg);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep the last incomplete part

            for (const line of lines) {
                if (!line.trim()) continue;

                const eventMatch = line.match(/^event: (.*)$/m);
                const dataMatch = line.match(/^data: (.*)$/m); // Using dotAll usually better but for single line json it works
                // Actually data can be multiline if I emitted it so, but I used json.dumps which escapes newlines for JSON.
                // For text data (status), it might be plain text.

                if (eventMatch && dataMatch) {
                    const eventType = eventMatch[1].trim();
                    const rawData = line.substring(line.indexOf('data: ') + 6);

                    try {
                        if (eventType === 'token') {
                            // Token is JSON string
                            onChunk(JSON.parse(rawData));
                        } else if (eventType === 'status') {
                            // Status is plain text
                            onStatus(rawData.trim());
                        } else if (eventType === 'metadata') {
                            onMetadata(JSON.parse(rawData));
                        } else if (eventType === 'error') {
                            onError(rawData);
                        }
                    } catch (e) {
                        console.error('Error parsing stream data:', e, line);
                    }
                }
            }
        }
    } catch (error) {
        onError(error.message);
    }
};

/**
 * Health check
 */
export const healthCheck = async () => {
    const response = await api.get('/health/');
    return response.data;
};

export default api;
