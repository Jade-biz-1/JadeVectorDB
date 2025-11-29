import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const ThreeDVisualization = ({ width = 600, height = 400 }) => {
  const mountRef = useRef(null);
  const [scene, setScene] = useState(null);
  const [camera, setCamera] = useState(null);
  const [renderer, setRenderer] = useState(null);
  const [controls, setControls] = useState(null);

  useEffect(() => {
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    setScene(scene);

    // Create camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    setCamera(camera);

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    setRenderer(renderer);

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Create sample 3D vector data
    const createVectorData = () => {
      const vectors = [];
      
      // Create central query vector
      vectors.push({
        position: [0, 0, 0],
        color: 0xef4444,
        size: 0.3,
        type: 'query'
      });
      
      // Create similar vectors in a sphere around the query
      for (let i = 0; i < 50; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const radius = 0.5 + Math.random() * 1.5;
        
        const x = radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.sin(phi) * Math.sin(theta);
        const z = radius * Math.cos(phi);
        
        vectors.push({
          position: [x, y, z],
          color: 0x3b82f6,
          size: 0.15 + Math.random() * 0.1,
          type: 'similar'
        });
      }
      
      // Create cluster vectors
      for (let cluster = 0; cluster < 3; cluster++) {
        const centerX = (cluster - 1) * 2;
        const centerY = (cluster % 2 - 0.5) * 2;
        const centerZ = (Math.floor(cluster / 2) - 0.5) * 2;
        const color = [0x8b5cf6, 0x10b981, 0xf59e0b][cluster];
        
        for (let i = 0; i < 30; i++) {
          const x = centerX + (Math.random() - 0.5) * 1.5;
          const y = centerY + (Math.random() - 0.5) * 1.5;
          const z = centerZ + (Math.random() - 0.5) * 1.5;
          
          vectors.push({
            position: [x, y, z],
            color: color,
            size: 0.1 + Math.random() * 0.05,
            type: 'cluster'
          });
        }
      }
      
      // Create background noise vectors
      for (let i = 0; i < 100; i++) {
        const x = (Math.random() - 0.5) * 10;
        const y = (Math.random() - 0.5) * 10;
        const z = (Math.random() - 0.5) * 10;
        
        vectors.push({
          position: [x, y, z],
          color: 0x94a3b8,
          size: 0.05 + Math.random() * 0.05,
          type: 'noise'
        });
      }
      
      return vectors;
    };

    // Add vector points to scene
    const vectorData = createVectorData();
    const vectorGroup = new THREE.Group();
    
    vectorData.forEach(vector => {
      const geometry = new THREE.SphereGeometry(vector.size, 16, 16);
      const material = new THREE.MeshPhongMaterial({ 
        color: vector.color,
        shininess: 30
      });
      
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(vector.position[0], vector.position[1], vector.position[2]);
      vectorGroup.add(sphere);
    });
    
    scene.add(vectorGroup);

    // Add connecting lines for similar vectors
    const queryVector = vectorData[0];
    for (let i = 1; i <= 50; i++) {
      const similarVector = vectorData[i];
      
      const lineGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(queryVector.position[0], queryVector.position[1], queryVector.position[2]),
        new THREE.Vector3(similarVector.position[0], similarVector.position[1], similarVector.position[2])
      ]);
      
      const lineMaterial = new THREE.LineBasicMaterial({ 
        color: 0x94a3b8,
        transparent: true,
        opacity: 0.5
      });
      
      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);
    }

    // Add coordinate axes
    const axesHelper = new THREE.AxesHelper(3);
    scene.add(axesHelper);

    // Add orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    setControls(controls);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate vector group slowly
      vectorGroup.rotation.y += 0.002;
      
      // Update controls
      if (controls) {
        controls.update();
      }
      
      // Render scene
      renderer.render(scene, camera);
    };
    
    animate();

    // Handle window resize
    const handleResize = () => {
      if (camera && renderer) {
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }
    };
    
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      
      if (mountRef.current && renderer) {
        mountRef.current.removeChild(renderer.domElement);
      }
      
      // Dispose of Three.js objects
      scene.traverse(object => {
        if (object.isMesh) {
          object.geometry.dispose();
          if (object.material.isMaterial) {
            object.material.dispose();
          }
        }
      });
      
      if (renderer) {
        renderer.dispose();
      }
    };
  }, [width, height]);

  return (
    <div className="w-full h-full relative">
      <div ref={mountRef} className="w-full h-full" style={{ height: '100%', minHeight: '400px' }} />
      
      <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded z-10">
        3D Vector Space
      </div>
      
      <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded z-10">
        Click and drag to rotate • Scroll to zoom
      </div>
    </div>
  );
};

export default ThreeDVisualization;