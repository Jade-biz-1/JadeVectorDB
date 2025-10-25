import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Share2, 
  Users, 
  Code,
  Heart,
  MessageCircle,
  Download,
  Upload,
  Eye,
  EyeOff,
  Copy,
  Check,
  Search,
  Star
} from "lucide-react";

const CommunitySharing = () => {
  const [activeTab, setActiveTab] = useState('share');
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [code, setCode] = useState('');
  const [tags, setTags] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState(null);
  const [showCode, setShowCode] = useState({});

  // Sample shared scenarios from community
  const [communityScenarios] = useState([
    {
      id: '1',
      title: 'Product Similarity Search',
      description: 'A tutorial scenario that demonstrates product similarity search using vector embeddings',
      author: 'vector_ninja',
      likes: 42,
      downloads: 128,
      tags: ['search', 'e-commerce', 'similarity'],
      code: `// Create a product database with 128-dimensional vectors
const db = await createDatabase({
  name: 'product_embeddings',
  dimensions: 128
});

// Add sample products
const products = [
  { id: 'p1', vector: [0.1, 0.5, 0.9, /* ... */], metadata: { category: 'shoes', price: 99.99 } },
  { id: 'p2', vector: [0.8, 0.3, 0.2, /* ... */], metadata: { category: 'shoes', price: 79.99 } },
  // ...
];

for (const product of products) {
  await addVector(db.id, product);
}

// Perform similarity search
const queryVector = [0.75, 0.45, 0.35, /* ... */];
const results = await similaritySearch(db.id, queryVector, { top_k: 5 });`,
      createdAt: '2023-10-10',
      stars: 4.7
    },
    {
      id: '2',
      title: 'Document Semantic Search',
      description: 'Semantic search for documents using vector embeddings',
      author: 'doc_reader',
      likes: 38,
      downloads: 96,
      tags: ['documents', 'search', 'metadata'],
      code: `// Create database for document embeddings
const db = await createDatabase({
  name: 'document_embeddings',
  dimensions: 512
});

// Add document embeddings
const documents = [
  { id: 'doc1', vector: [/* embedding values */], metadata: { title: 'AI Research Paper', category: 'research' } },
  // ...
];

for (const doc of documents) {
  await addVector(db.id, doc);
}

// Semantic search
const query = "machine learning algorithms";
const queryEmbedding = await getEmbedding(query);
const results = await similaritySearch(db.id, queryEmbedding, { 
  top_k: 10,
  filters: { category: 'research' }
});`,
      createdAt: '2023-10-12',
      stars: 4.5
    },
    {
      id: '3',
      title: 'Image Similarity Engine',
      description: 'Find similar images using visual embeddings',
      author: 'vision_dev',
      likes: 67,
      downloads: 187,
      tags: ['images', 'similarity', 'computer-vision'],
      code: `// Create image embedding database
const db = await createDatabase({
  name: 'image_embeddings',
  dimensions: 2048
});

// Add image embeddings
const images = [
  { id: 'img1', vector: [/* visual embedding */], metadata: { path: '/path/to/image1.jpg', tags: ['cat', 'pet'] } },
  // ...
];

for (const image of images) {
  await addVector(db.id, image);
}

// Find similar images
const targetImageEmbedding = await getEmbeddingFromImage('/path/to/target.jpg');
const results = await similaritySearch(db.id, targetImageEmbedding, { 
  top_k: 10,
  threshold: 0.8 
});`,
      createdAt: '2023-10-15',
      stars: 4.9
    }
  ]);

  const [mySharedScenarios, setMySharedScenarios] = useState([
    {
      id: 'my1',
      title: 'Custom Search Algorithm',
      description: 'My custom search algorithm with special filtering',
      likes: 12,
      downloads: 34,
      tags: ['search', 'custom', 'filtering'],
      createdAt: '2023-10-05',
      stars: 4.2
    }
  ]);

  const handleShare = () => {
    if (title.trim() && code.trim()) {
      // In a real implementation, this would upload the scenario to a backend
      const newScenario = {
        id: `new_${Date.now()}`,
        title,
        description,
        author: 'current_user', // Would be from authentication
        likes: 0,
        downloads: 0,
        tags: tags.split(',').map(tag => tag.trim()).filter(tag => tag),
        code,
        createdAt: new Date().toISOString().split('T')[0],
        stars: 0
      };
      
      setMySharedScenarios([...mySharedScenarios, newScenario]);
      
      // Reset form
      setTitle('');
      setDescription('');
      setCode('');
      setTags('');
      
      setActiveTab('my-shared');
    }
  };

  const handleLike = (id, section) => {
    // In a real implementation, this would update the backend
    if (section === 'community') {
      // Update communityScenarios with new like count
    } else if (section === 'my-shared') {
      // Update mySharedScenarios with new like count
    }
  };

  const handleDownload = (id) => {
    // In a real implementation, this would download the scenario
    console.log(`Downloading scenario ${id}`);
  };

  const copyCodeToClipboard = (code, id) => {
    navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleCodeVisibility = (id) => {
    setShowCode(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const filteredCommunityScenarios = communityScenarios.filter(scenario => 
    scenario.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    scenario.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    scenario.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <Card className="h-full overflow-hidden flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Share2 className="h-5 w-5" />
          Community Sharing
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Share your scenarios or explore what others have created
        </p>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="share">
              <Upload className="h-3 w-3 mr-1" />
              Share
            </TabsTrigger>
            <TabsTrigger value="community">
              <Users className="h-3 w-3 mr-1" />
              Community
            </TabsTrigger>
            <TabsTrigger value="my-shared">
              <Code className="h-3 w-3 mr-1" />
              My Shared
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="share" className="flex-1 flex flex-col overflow-hidden">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Title</label>
                <Input 
                  value={title} 
                  onChange={(e) => setTitle(e.target.value)} 
                  placeholder="Enter a descriptive title for your scenario" 
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Description</label>
                <Textarea 
                  value={description} 
                  onChange={(e) => setDescription(e.target.value)} 
                  placeholder="Describe what your scenario does and how it works" 
                  rows={3}
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Tags</label>
                <Input 
                  value={tags} 
                  onChange={(e) => setTags(e.target.value)} 
                  placeholder="Enter tags separated by commas (e.g. search, similarity, embedding)" 
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Code Snippet</label>
                <div className="relative">
                  <Textarea 
                    value={code} 
                    onChange={(e) => setCode(e.target.value)} 
                    placeholder="Paste your code here..." 
                    rows={8}
                    className="font-mono text-sm"
                  />
                </div>
              </div>
              
              <div className="flex justify-end">
                <Button onClick={handleShare}>
                  <Upload className="h-4 w-4 mr-2" />
                  Share Scenario
                </Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="community" className="flex-1 flex flex-col overflow-hidden">
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input 
                  value={searchQuery} 
                  onChange={(e) => setSearchQuery(e.target.value)} 
                  placeholder="Search scenarios by title, description, or tags..." 
                  className="pl-8"
                />
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto space-y-4">
              {filteredCommunityScenarios.map(scenario => (
                <Card key={scenario.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-semibold">{scenario.title}</h3>
                        <p className="text-sm text-muted-foreground mt-1">{scenario.description}</p>
                        <div className="flex items-center gap-4 mt-2">
                          <span className="text-xs text-muted-foreground">by {scenario.author}</span>
                          <div className="flex items-center gap-1">
                            {[...Array(5)].map((_, i) => (
                              <Star 
                                key={i} 
                                className={`h-3 w-3 ${i < Math.floor(scenario.stars) ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} 
                              />
                            ))}
                            <span className="text-xs ml-1">{scenario.stars}</span>
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-1 mt-2">
                          {scenario.tags.map((tag, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex flex-col items-end">
                        <span className="text-xs text-muted-foreground">{scenario.createdAt}</span>
                        <div className="flex items-center gap-2 mt-2">
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => handleLike(scenario.id, 'community')}
                          >
                            <Heart className="h-3 w-3 mr-1" />
                            {scenario.likes}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => handleDownload(scenario.id)}
                          >
                            <Download className="h-3 w-3 mr-1" />
                            {scenario.downloads}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => toggleCodeVisibility(scenario.id)}
                          >
                            {showCode[scenario.id] ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                          </Button>
                        </div>
                      </div>
                    </div>
                    
                    {showCode[scenario.id] && (
                      <div className="mt-4 relative">
                        <pre className="bg-muted rounded-md p-4 text-xs overflow-x-auto">
                          <code>{scenario.code}</code>
                        </pre>
                        <Button 
                          size="sm" 
                          variant="secondary" 
                          onClick={() => copyCodeToClipboard(scenario.code, scenario.id)}
                          className="absolute top-2 right-2"
                        >
                          {copiedId === scenario.id ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
              
              {filteredCommunityScenarios.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  No scenarios found matching your search.
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="my-shared" className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto space-y-4">
              {mySharedScenarios.map(scenario => (
                <Card key={scenario.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-semibold">{scenario.title}</h3>
                        <p className="text-sm text-muted-foreground mt-1">{scenario.description}</p>
                        
                        <div className="flex items-center gap-4 mt-2">
                          <div className="flex items-center gap-1">
                            {[...Array(5)].map((_, i) => (
                              <Star 
                                key={i} 
                                className={`h-3 w-3 ${i < Math.floor(scenario.stars) ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} 
                              />
                            ))}
                            <span className="text-xs ml-1">{scenario.stars}</span>
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-1 mt-2">
                          {scenario.tags.map((tag, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex flex-col items-end">
                        <span className="text-xs text-muted-foreground">{scenario.createdAt}</span>
                        <div className="flex items-center gap-2 mt-2">
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => handleLike(scenario.id, 'my-shared')}
                          >
                            <Heart className="h-3 w-3 mr-1" />
                            {scenario.likes}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => handleDownload(scenario.id)}
                          >
                            <Download className="h-3 w-3 mr-1" />
                            {scenario.downloads}
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
              
              {mySharedScenarios.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  You haven't shared any scenarios yet. Create one in the Share tab!
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default CommunitySharing;