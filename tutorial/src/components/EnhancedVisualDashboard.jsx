import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useTutorial } from '../contexts/TutorialContext';

const EnhancedVisualDashboard = () => {
  const svgRef = useRef();
  const [visualizationType, setVisualizationType] = useState('2d-tsne');
  const [similarityMetric, setSimilarityMetric] = useState('cosine');
  const [sampleSize, setSampleSize] = useState(1000);
  const { tutorialState } = useTutorial();

  useEffect(() => {
    // Clear previous drawing
    d3.select(svgRef.current).selectAll("*").remove();
    
    const svg = d3.select(svgRef.current);
    const width = 800;
    const height = 400;
    
    // Add background
    svg.append("rect")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("fill", "#f9fafb");
    
    // Create sample data points with more realistic distribution
    const generateSampleData = () => {
      const data = [];
      const colors = ["#ef4444", "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"];
      
      // Generate query vector
      data.push({
        id: "query-vector",
        x: width * 0.5,
        y: height * 0.5,
        label: "Query Vector",
        color: colors[0],
        size: 10,
        type: "query"
      });
      
      // Generate similar vectors (closer to query)
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * 2 * Math.PI;
        const distance = 50 + Math.random() * 30;
        const x = width * 0.5 + Math.cos(angle) * distance;
        const y = height * 0.5 + Math.sin(angle) * distance;
        
        data.push({
          id: `similar-${i}`,
          x: x,
          y: y,
          label: `Similar ${i+1}`,
          color: colors[1],
          size: 7,
          type: "similar",
          similarity: (0.9 - i * 0.05).toFixed(2)
        });
      }
      
      // Generate cluster vectors
      for (let cluster = 0; cluster < 3; cluster++) {
        const centerX = 150 + cluster * 250;
        const centerY = 100 + (cluster % 2) * 200;
        const clusterColor = colors[2 + cluster];
        
        for (let i = 0; i < 12; i++) {
          const angle = (i / 12) * 2 * Math.PI;
          const distance = 30 + Math.random() * 20;
          const x = centerX + Math.cos(angle) * distance;
          const y = centerY + Math.sin(angle) * distance;
          
          data.push({
            id: `cluster-${cluster}-${i}`,
            x: x,
            y: y,
            label: `Cluster ${cluster+1}.${i+1}`,
            color: clusterColor,
            size: 5,
            type: "cluster"
          });
        }
      }
      
      // Generate random background vectors
      for (let i = 0; i < 20; i++) {
        data.push({
          id: `random-${i}`,
          x: Math.random() * width,
          y: Math.random() * height,
          label: `Random ${i+1}`,
          color: colors[4],
          size: 4,
          type: "random"
        });
      }
      
      return data;
    };
    
    const data = generateSampleData();
    
    // Draw data points
    const circles = svg.selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", d => d.size)
      .attr("fill", d => d.color)
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 1.5)
      .attr("class", "cursor-pointer")
      .on("mouseover", function(event, d) {
        d3.select(this).attr("r", d.size + 2);
        // Show tooltip
        const tooltip = svg.append("g")
          .attr("class", "tooltip")
          .attr("transform", `translate(${d.x + 15}, ${d.y - 15})`);
        
        tooltip.append("rect")
          .attr("rx", 4)
          .attr("ry", 4)
          .attr("width", 120)
          .attr("height", d.similarity ? 60 : 40)
          .attr("fill", "rgba(0, 0, 0, 0.8)")
          .attr("stroke", "#ffffff")
          .attr("stroke-width", 1);
        
        tooltip.append("text")
          .attr("x", 8)
          .attr("y", 18)
          .attr("fill", "#ffffff")
          .attr("font-size", "12px")
          .attr("font-weight", "bold")
          .text(d.label);
        
        tooltip.append("text")
          .attr("x", 8)
          .attr("y", 34)
          .attr("fill", "#cccccc")
          .attr("font-size", "11px")
          .text(`Type: ${d.type}`);
        
        if (d.similarity) {
          tooltip.append("text")
            .attr("x", 8)
            .attr("y", 50)
            .attr("fill", "#ffff00")
            .attr("font-size", "11px")
            .text(`Similarity: ${d.similarity}`);
        }
      })
      .on("mouseout", function() {
        d3.select(this).attr("r", d => d.size);
        svg.selectAll(".tooltip").remove();
      });
    
    // Add labels to important points
    const labeledPoints = data.filter(d => d.type === "query" || d.type === "similar");
    svg.selectAll("text.label")
      .data(labeledPoints)
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("x", d => d.x + 15)
      .attr("y", d => d.y - 10)
      .text(d => d.label)
      .attr("font-size", "12px")
      .attr("font-weight", d => d.type === "query" ? "bold" : "normal")
      .attr("fill", d => d.type === "query" ? "#dc2626" : "#374151");
    
    // Draw similarity lines (between query and similar vectors)
    const similarVectors = data.filter(d => d.type === "similar");
    similarVectors.forEach((vector, i) => {
      const queryVector = data.find(d => d.type === "query");
      if (queryVector) {
        svg.append("line")
          .attr("x1", queryVector.x)
          .attr("y1", queryVector.y)
          .attr("x2", vector.x)
          .attr("y2", vector.y)
          .attr("stroke", "#9ca3af")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "3,3");
        
        // Add similarity label
        if (vector.similarity) {
          const midX = (queryVector.x + vector.x) / 2;
          const midY = (queryVector.y + vector.y) / 2;
          
          svg.append("text")
            .attr("x", midX + 10)
            .attr("y", midY - 10)
            .text(vector.similarity)
            .attr("font-size", "10px")
            .attr("fill", "#6b7280")
            .attr("font-weight", "bold");
        }
      }
    });
    
    // Add legend
    const legend = svg.append("g")
      .attr("transform", "translate(20, 20)");
    
    const legendItems = [
      { label: "Query Vector", color: "#ef4444" },
      { label: "Similar Vectors", color: "#3b82f6" },
      { label: "Cluster Vectors", color: "#8b5cf6" },
      { label: "Random Vectors", color: "#f59e0b" }
    ];
    
    legendItems.forEach((item, i) => {
      const group = legend.append("g")
        .attr("transform", `translate(0, ${i * 25})`);
      
      group.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", 6)
        .attr("fill", item.color);
      
      group.append("text")
        .attr("x", 15)
        .attr("y", 4)
        .text(item.label)
        .attr("font-size", "12px")
        .attr("fill", "#374151");
    });
    
    // Add axes for better orientation
    const xAxis = d3.axisBottom(d3.scaleLinear().domain([0, width]).range([0, width]))
      .ticks(5);
    
    const yAxis = d3.axisLeft(d3.scaleLinear().domain([0, height]).range([0, height]))
      .ticks(5);
    
    svg.append("g")
      .attr("transform", `translate(0, ${height - 20})`)
      .call(xAxis)
      .attr("font-size", "10px")
      .attr("color", "#9ca3af");
    
    svg.append("g")
      .attr("transform", "translate(20, 0)")
      .call(yAxis)
      .attr("font-size", "10px")
      .attr("color", "#9ca3af");
    
  }, [visualizationType, similarityMetric, sampleSize]);

  return (
    <div className="module-card">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800 mb-2 md:mb-0">Vector Space Visualization</h2>
        
        <div className="flex flex-wrap gap-2">
          <div className="flex items-center">
            <label className="mr-2 text-sm text-gray-700">View:</label>
            <select 
              value={visualizationType} 
              onChange={(e) => setVisualizationType(e.target.value)}
              className="p-1 border border-gray-300 rounded text-sm"
            >
              <option value="2d-tsne">2D t-SNE</option>
              <option value="2d-pca">2D PCA</option>
              <option value="3d">3D View</option>
            </select>
          </div>
          
          <div className="flex items-center">
            <label className="mr-2 text-sm text-gray-700">Metric:</label>
            <select 
              value={similarityMetric} 
              onChange={(e) => setSimilarityMetric(e.target.value)}
              className="p-1 border border-gray-300 rounded text-sm"
            >
              <option value="cosine">Cosine</option>
              <option value="euclidean">Euclidean</option>
              <option value="dot">Dot Product</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          <div className="border border-gray-200 rounded-lg bg-white p-4">
            <div className="visualization-container relative">
              <svg 
                ref={svgRef} 
                width="100%" 
                height="400" 
                viewBox="0 0 800 400"
                className="w-full border border-gray-100 rounded"
              />
              
              <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                {sampleSize.toLocaleString()} vectors
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
                <div className="text-2xl font-bold text-blue-800">1,000</div>
                <div className="text-sm text-blue-700">Vectors</div>
              </div>
              <div className="bg-green-50 p-3 rounded-lg border border-green-100">
                <div className="text-2xl font-bold text-green-800">0.85</div>
                <div className="text-sm text-green-700">Avg Similarity</div>
              </div>
              <div className="bg-purple-50 p-3 rounded-lg border border-purple-100">
                <div className="text-2xl font-bold text-purple-800">128</div>
                <div className="text-sm text-purple-700">Dimensions</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="lg:w-80">
          <div className="bg-white rounded-lg border border-gray-200 p-4 h-full">
            <h3 className="font-medium text-gray-800 mb-3">Visualization Controls</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sample Size</label>
                <input 
                  type="range" 
                  min="100" 
                  max="10000" 
                  value={sampleSize} 
                  onChange={(e) => setSampleSize(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>100</span>
                  <span className="font-medium">{sampleSize.toLocaleString()}</span>
                  <span>10,000</span>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Color Scheme</label>
                <select className="w-full p-2 border border-gray-300 rounded-md text-sm">
                  <option>Default</option>
                  <option>Cluster-based</option>
                  <option>Similarity-based</option>
                  <option>Category-based</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Point Size</label>
                <select className="w-full p-2 border border-gray-300 rounded-md text-sm">
                  <option>Fixed</option>
                  <option>By Similarity</option>
                  <option>By Magnitude</option>
                </select>
              </div>
              
              <div className="pt-2">
                <button className="w-full btn-secondary text-sm">
                  Export Visualization
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="font-medium text-gray-800 mb-2">About This Visualization</h3>
        <p className="text-sm text-gray-600">
          This 2D projection shows how vectors are distributed in high-dimensional space. 
          The red query vector is at the center, with blue vectors representing similar items. 
          Purple clusters show groups of related vectors. Adjust the controls to explore different 
          visualization techniques and metrics.
        </p>
      </div>
    </div>
  );
};

export default EnhancedVisualDashboard;