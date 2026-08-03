import os

def create_drawio():
    xml_content = """<mxfile host="65bd71144e">
    <diagram id="AdaptiveSR_Architecture" name="AdaptiveSR End-to-End Flow">
        <mxGraphModel dx="1200" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
                
                <!-- Row 0: Input Source -->
                <mxCell id="2" value="&lt;b&gt;Input Source&lt;/b&gt;&lt;br&gt;• Local Video&lt;br&gt;• YouTube URL&lt;br&gt;• Live Stream" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="320" y="40" width="180" height="80" as="geometry"/>
                </mxCell>
                
                <!-- Row 1: Input Acquisition Layer -->
                <mxCell id="3" value="&lt;b&gt;Input Acquisition Layer&lt;/b&gt;&lt;br&gt;(Download / Decode)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;align=center;" vertex="1" parent="1">
                    <mxGeometry x="320" y="160" width="180" height="60" as="geometry"/>
                </mxCell>
                
                <!-- Row 2: Monitors / Analyzer -->
                <mxCell id="4" value="&lt;b&gt;Network Monitor&lt;/b&gt;&lt;br&gt;• Bandwidth&lt;br&gt;• Bitrate&lt;br&gt;• Latency&lt;br&gt;• Connection Type" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="80" y="270" width="170" height="110" as="geometry"/>
                </mxCell>
                <mxCell id="5" value="&lt;b&gt;Device Monitor&lt;/b&gt;&lt;br&gt;• CPU Usage&lt;br&gt;• GPU Usage&lt;br&gt;• RAM Usage&lt;br&gt;• Battery&lt;br&gt;• Temperature" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="325" y="270" width="170" height="110" as="geometry"/>
                </mxCell>
                <mxCell id="6" value="&lt;b&gt;Scene Analyzer&lt;/b&gt;&lt;br&gt;• Motion&lt;br&gt;• Texture&lt;br&gt;• Complexity&lt;br&gt;• Edge Density&lt;br&gt;• Semantic Score" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="570" y="270" width="170" height="110" as="geometry"/>
                </mxCell>
                
                <!-- Row 3: Adaptive Decision Engine -->
                <mxCell id="7" value="&lt;b&gt;Adaptive Decision Engine&lt;/b&gt;&lt;br&gt;Multi-Criteria Decision Making&lt;br&gt;(QoE + Resource Optimization)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=center;" vertex="1" parent="1">
                    <mxGeometry x="290" y="430" width="240" height="70" as="geometry"/>
                </mxCell>
                
                <!-- Row 4: Selectors -->
                <mxCell id="8" value="&lt;b&gt;Bitrate Selector&lt;/b&gt;&lt;br&gt;• 480p / 720p&lt;br&gt;• 1080p&lt;br&gt;• 4K" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="80" y="550" width="170" height="90" as="geometry"/>
                </mxCell>
                <mxCell id="9" value="&lt;b&gt;Model Selector&lt;/b&gt;&lt;br&gt;• BasicVSR++&lt;br&gt;• Real-ESRGAN&lt;br&gt;• TinySR" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="325" y="550" width="170" height="90" as="geometry"/>
                </mxCell>
                <mxCell id="10" value="&lt;b&gt;Scale Selector&lt;/b&gt;&lt;br&gt;• ×2&lt;br&gt;• ×4&lt;br&gt;• Native" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;align=left;spacingLeft=15;" vertex="1" parent="1">
                    <mxGeometry x="570" y="550" width="170" height="90" as="geometry"/>
                </mxCell>
                
                <!-- Row 5: VSR Engine -->
                <mxCell id="11" value="&lt;b&gt;Video Super-Resolution Engine&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;align=center;" vertex="1" parent="1">
                    <mxGeometry x="290" y="690" width="240" height="50" as="geometry"/>
                </mxCell>
                
                <!-- Row 6: Reconstruction & Post-processing -->
                <mxCell id="12" value="&lt;b&gt;Reconstruction &amp; Post-processing&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;align=center;" vertex="1" parent="1">
                    <mxGeometry x="290" y="790" width="240" height="50" as="geometry"/>
                </mxCell>
                
                <!-- Row 7: Enhanced Output Video -->
                <mxCell id="13" value="&lt;b&gt;Enhanced Output Video&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=center;" vertex="1" parent="1">
                    <mxGeometry x="290" y="890" width="240" height="50" as="geometry"/>
                </mxCell>
                
                <!-- Connections Row 0 -> Row 1 -->
                <mxCell id="14" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="2" target="3">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                
                <!-- Connections Row 1 -> Row 2 -->
                <mxCell id="15" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="3" target="4">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="16" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="3" target="5">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="17" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="3" target="6">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                
                <!-- Connections Row 2 -> Row 3 -->
                <mxCell id="18" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="4" target="7">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="19" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="5" target="7">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="20" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="6" target="7">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                
                <!-- Connections Row 3 -> Row 4 -->
                <mxCell id="21" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="7" target="8">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="22" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="7" target="9">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="23" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="7" target="10">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                
                <!-- Connections Row 4 -> Row 5 -->
                <mxCell id="24" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="8" target="11">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="25" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="9" target="11">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="26" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="10" target="11">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                
                <!-- Connections Row 5 -> Row 6 -> Row 7 -->
                <mxCell id="27" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="11" target="12">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="28" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="12" target="13">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
            </root>
        </mxGraphModel>
    </diagram>
</mxfile>
"""
    with open("architecture.drawio", "w", encoding="utf-8") as f:
        f.write(xml_content)
    print("architecture.drawio compiled successfully with the requested diagram!")

if __name__ == "__main__":
    create_drawio()
