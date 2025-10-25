// API route for recording requests
export default async function handler(req, res) {
  if (req.method === 'POST') {
    try {
      const { sessionId } = req.body;
      
      // Simulate recording a request
      return res.status(200).json({ success: true, recordedAt: new Date().toISOString() });
    } catch (error) {
      return res.status(500).json({ error: 'Internal server error' });
    }
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}