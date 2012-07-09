/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Robert Bosch LLC.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Robert Bosch nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

ros.tf.TransformListener = Class.extend(
/** @lends ros.tf.TransformListener# */
{

	init: function(node, tf_topic){
		// members
		this.tf_prefix = "";
		this.tree = null;
		this.numTransforms = 0;
		this.node = node;
		this.transformLookup = new ros.Map();
  
		// subscribe to tf
		var that = this;
		this.node.subscribe(tf_topic,function(msg){that.updateFromMessage(msg);});
	},

	findRootNode : function (transforms)
	{
		//find root node
		var frames = new Array();
		var childFrames = new Array();
		
		for ( var t in transforms) {
			var transform = transforms[t];
    
			if (frames.indexOf(transform.frame_id) == -1)
			{
				frames.push(transform.frame_id);
			}
    
			if (frames.indexOf(transform.child_frame_id) == -1)
			{
				frames.push(transform.child_frame_id);
			}

			if (childFrames.indexOf(transform.child_frame_id) == -1)
			{
				childFrames.push(transform.child_frame_id);
			}	
		}
  
		// check if frames is in child frames
		for ( var f in frames) {
			var frame = frames[f];
			// found the root if frame is not a child frame
			if (childFrames.indexOf(frame) == -1)
			{
				return frame;
			}	
		}
  
	},

	buildTransformTree : function (transforms)
	{
		var tree = new ros.tf.TransformTree();
  
		//find root node id
		var root_frame_id = this.findRootNode(transforms);
		var rootNode = new ros.tf.TransformTreeNode(root_frame_id);
  
		tree.setRootNode(rootNode);
  
		// build transform tree from transforms
		this.addTreeNodeCildrenRecursive(transforms, rootNode);
  
		return tree;
	},

	addTreeNodeCildrenRecursive : function (transforms, node)
	{
		for ( var t in transforms) {
			var transform = transforms[t];
			if(transform.frame_id == node.frame_id) {
				// create a new child
				var child = new ros.tf.TransformTreeNode(transform.child_frame_id);
				var matrix = transform.matrix;
				child.parent = node;
				node.transforms.push(matrix);
				node.children.push(child);
      
				// build transform tree recursively
				this.addTreeNodeCildrenRecursive(transforms, child);
			}
		}
	},

	updateTransformTree : function (transforms)
	{
		for ( var t in transforms) {
			var transform = transforms[t];
			this.updateTransform(transform);
		}
	},

	updateFromMessage : function (tf_msg)
	{  	 
		// create a list of stamped transforms
		var transforms = new Array();
		for ( var t in tf_msg.transforms) {
			var transform_msg = tf_msg.transforms[t];
			var transform = new ros.tf.StampedTransform();
			transform.updateFromMessage(transform_msg);
			transforms.push(transform);
		}
		
		if(!this.tree || this.numTransforms < transforms.length) {
			// build a tree from transforms
			var tree = this.buildTransformTree(transforms);
			this.tree = tree;
			this.numTransforms = transforms.length;
		}
		else {
			// update the tree
			this.updateTransformTree(transforms);
		}

		this.updateTransformLookupTable();
	},

	updateTransform : function (transform)
	{	
		//ros_debug("tf: updating " + transform.frame_id + "<--" + transform.child_frame_id);
		transform.frame_id = ros.tf.formatFrameID(transform.frame_id);
		transform.child_frame_id = ros.tf.formatFrameID(transform.child_frame_id);
  
		// get a list of all nodes
		var foundNode = false;
		var nodes = this.tree.toList();
		for (var n in nodes) {
			var node = nodes[n];
			if(node.frame_id == transform.frame_id) {
				foundNode = true;
				var childID = node.getChildID(transform.child_frame_id);
				if(childID != -1) {
					node.transforms[childID] = transform.matrix;
				}
				else
				{
					// create a new child
					var child = new ros.tf.TransformTreeNode(transform.child_frame_id);
					var matrix = transform.matrix;
					child.parent = node;
					node.transforms.push(matrix);
					node.children.push(child);
					ros_debug("tf: adding " + child.frame_id + " as new child node");
				}
				// we can stop here
				return;
			}
		}
  
		// if no node was found try to add new root node
		if(transform.child_frame_id == this.tree.rootNode.frame_id) {
			// create new root node
			var child = this.tree.rootNode;
			var rootNode = new ros.tf.TransformTreeNode(transform.frame_id);
			var matrix = transform.matrix;
			child.parent = rootNode;
			rootNode.transforms.push(matrix);
			rootNode.children.push(child);
			this.tree.rootNode = rootNode;
			ros_debug("tf: adding " + transform.frame_id + " as new root node");
			// we can stop here
			return;
		}
		ros_debug("tf: coudln't update transform " + transform.frame_id + "<---" + transform.child_frame_id);
	},

	updateTransformLookupTable : function()
	{
		var mat = new SglMatrixStack();
		var map = new ros.Map();
		var root_node = this.tree.getRootNode();

		if(root_node) {
			this.updateTransformLookupTableRecursive(map,mat,root_node);
			var matrix = mat.top;
			map.insert(root_node.frame_id,matrix);
			mat.pop();
			
			this.transformLookup = map;
		}
	},

	updateTransformLookupTableRecursive : function(map, mat, parent_node)
	{
		for(var n in parent_node.children) {
			var node = parent_node.children[n]; 
			var transform = parent_node.transforms[n];
			mat.push();
			mat.multiply(transform);
			this.updateTransformLookupTableRecursive(map,mat,node);
			var matrix = mat.top;
			map.insert(node.frame_id,matrix);
			mat.pop();
		}
	},

	lookupTransformRecursive :  function (stack, node, matrix, frame_id)
	{
		stack.push();
		stack.multiply(matrix);
		if(frame_id == node.frame_id) {
			return stack.top;
		}
		// check children
		for ( var c in node.children) {
			var matrix = this.lookupTransformRecursive(stack, node.children[c], node.transforms[c], frame_id);
			if(matrix) {
				return matrix;
			}
		}
		stack.pop(); 
		return null;
	},

	lookupTransformMartix : function (target_frame, source_frame)
	{
		var rootNode = this.tree.getRootNode();
		if(!rootNode) {
			return null;
		}

		//find source frame transform 
		var source_transform = this.transformLookup.find(source_frame);
		//  var source_transform = this.lookupTransformRecursive(stack, rootNode, sglIdentityM4(), source_frame);
		if(!source_transform) {
			return null;
		}	
		//ros.dumpMatrix("source_transform",source_transform,4,4);
		
		//find target frame transform
		var target_transform = this.transformLookup.find(target_frame);
		//  var target_transform = this.lookupTransformRecursive(stack, rootNode, sglIdentityM4(), target_frame);
		if(!target_transform) {
			return null;
		}
  
		var source_inverse = sglInverseM4(source_transform);
		return sglMulM4(source_inverse,target_transform);
	},

	lookupTransform : function (target_frame, source_frame)
	{
		var matrix = this.lookupTransformMartix(target_frame, source_frame);
		if(!matrix) {
			return null;
		}
		var transform = new ros.math.Transform();  
		transform.fromMatrix(matrix);
		return transform;
	},
	
});



