// API route for resetting sessions
export default async function handler(req, res) {
  if (req.method === 'POST') {
    try {
      const { sessionId } = req.body;
      
      // Simulate resetting a session
      return res.status(200).json({ success: true, message: 'Session reset successfully' });
    } catch (error) {
      return res.status(500).json({ error: 'Internal server error' });
    }
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}