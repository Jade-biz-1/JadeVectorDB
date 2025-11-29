// API route for getting resource usage
export default async function handler(req, res) {
  if (req.method === 'POST') {
    try {
      const { sessionId } = req.body;
      
      // Simulate getting resource usage
      const mockUsage = {
        api_calls_made: Math.floor(Math.random() * 60),
        vectors_stored: Math.floor(Math.random() * 1000),
        databases_created: Math.floor(Math.random() * 10),
        memory_used_bytes: Math.floor(Math.random() * 100 * 1024 * 1024)
      };
      
      return res.status(200).json(mockUsage);
    } catch (error) {
      return res.status(500).json({ error: 'Internal server error' });
    }
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}