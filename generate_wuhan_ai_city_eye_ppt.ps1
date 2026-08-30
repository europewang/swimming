param(
    [string]$OutputPath = "D:\myflie\ALL_CODE\swimming\武汉城市智眼_AI科普_3到4分钟.pptx"
)

$ErrorActionPreference = "Stop"

function RGBColor([int]$r, [int]$g, [int]$b) {
    return $r + ($g * 256) + ($b * 65536)
}

function AddTextBox($slide, $x, $y, $w, $h, $text, $fontSize, $color, $fontName = "Microsoft YaHei UI", $bold = 0) {
    $shape = $slide.Shapes.AddTextbox(1, [float]$x, [float]$y, [float]$w, [float]$h)
    $shape.TextFrame.TextRange.Text = [string]$text
    $shape.TextFrame.TextRange.Font.Name = [string]$fontName
    $shape.TextFrame.TextRange.Font.Size = [float]$fontSize
    $shape.TextFrame.TextRange.Font.Color.RGB = [int]$color
    $shape.TextFrame.TextRange.Font.Bold = [int]$bold
    return $shape
}

function AddRect($slide, [int]$shapeType, [float]$x, [float]$y, [float]$w, [float]$h, [int]$fillColor, [float]$transparency = 0, [int]$lineColor = -1, [float]$lineWeight = 0.75) {
    $shape = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    $shape.Fill.Visible = -1
    $shape.Fill.ForeColor.RGB = $fillColor
    $shape.Fill.Transparency = $transparency
    if ($lineColor -ge 0) {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = $lineColor
        $shape.Line.Weight = $lineWeight
    } else {
        $shape.Line.Visible = 0
    }
    return $shape
}

function AddCard($slide, [float]$x, [float]$y, [float]$w, [float]$h, [int]$fillColor, [string]$title, [string]$body, [int]$accentColor) {
    $card = AddRect $slide 5 $x $y $w $h $fillColor 0.06
    $titleShape = AddTextBox $slide ($x + 18) ($y + 14) ($w - 36) 24 $title 20 (RGBColor 255 255 255) "Microsoft YaHei UI" -1
    $bodyShape = AddTextBox $slide ($x + 18) ($y + 46) ($w - 36) ($h - 60) $body 11 (RGBColor 224 232 241) "Microsoft YaHei UI" 0
    $bar = AddRect $slide 1 ($x + 18) ($y + $h - 14) 44 4 $accentColor 0
    return @($card, $titleShape, $bodyShape, $bar)
}

function AddPictureSafe($slide, [string]$path, [float]$x, [float]$y, [float]$w, [float]$h) {
    if (Test-Path $path) {
        return $slide.Shapes.AddPicture($path, 0, -1, $x, $y, $w, $h)
    }
    return $null
}

function NewSlide($presentation, [int]$index) {
    [void]$presentation.Slides.Add($index, 12)
    return $presentation.Slides.Item($index)
}

$workspace = "D:\myflie\ALL_CODE\swimming"
$assetDir = Join-Path $workspace "ppt_assets"
if (-not (Test-Path $assetDir)) {
    New-Item -ItemType Directory -Path $assetDir | Out-Null
}

$coverImage = Join-Path $assetDir "wuhan_cover.png"
$streetImage = Join-Path $assetDir "wuhan_street_ai.png"

$pp = $null
$presentation = $null

try {
    $pp = New-Object -ComObject PowerPoint.Application
    $pp.Visible = -1

    $presentation = $pp.Presentations.Add($true)
    Start-Sleep -Seconds 2
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    $cBg = RGBColor 7 18 33
    $cNavy = RGBColor 14 36 66
    $cBlue = RGBColor 35 95 201
    $cCyan = RGBColor 54 198 211
    $cMint = RGBColor 92 230 190
    $cGold = RGBColor 255 191 71
    $cWhite = RGBColor 245 248 252
    $cSoft = RGBColor 195 208 224
    $cRed = RGBColor 233 105 104
    $cCard = RGBColor 17 31 53

    # Slide 1
    $slide = NewSlide $presentation 1
    AddRect $slide 1 0 0 960 540 $cBg 0 | Out-Null
    AddPictureSafe $slide $coverImage 0 0 960 540 | Out-Null
    AddRect $slide 1 0 0 960 540 (RGBColor 5 14 28) 0.28 | Out-Null
    AddRect $slide 5 42 40 130 28 $cCyan 0.14 $cCyan 0.8 | Out-Null
    $tag = AddTextBox $slide 58 47 110 18 "讲科学·科普演讲" 11 $cWhite "Microsoft YaHei UI" -1
    $title = AddTextBox $slide 52 126 560 120 "《算法之下：AI如何重塑人类生活》" 28 $cWhite "Microsoft YaHei UI" -1
    $subtitle = AddTextBox $slide 54 250 520 78 '从武汉“城市智眼”出发，看AI算法如何与测绘地理信息技术一起，改变我们的出行、治理与城市规划。' 15 $cWhite "Microsoft YaHei UI" 0
    $footer = AddTextBox $slide 54 450 420 28 "小切口：一个普通路口    大逻辑：一座城市的数字感知系统" 12 $cSoft "Microsoft YaHei UI" 0

    # Slide 2
    $slide = NewSlide $presentation 2
    AddRect $slide 1 0 0 960 540 (RGBColor 248 250 253) 0 | Out-Null
    AddRect $slide 1 0 0 960 92 $cNavy 0 | Out-Null
    AddTextBox $slide 52 26 500 34 "1. 从一个普通路口开始" 24 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 52 112 390 52 "早高峰走过斑马线时，你以为自己只是路过；\n而城市系统正在同时观察交通、设施、安全与服务需求。" 17 $cNavy "Microsoft YaHei UI" 0 | Out-Null
    AddPictureSafe $slide $streetImage 518 112 388 232 | Out-Null
    AddCard $slide 52 336 258 142 (RGBColor 27 82 180) "我们看到的是生活" "红绿灯、公交进站、共享单车停放、行人穿行，这些都是每天最普通的城市切片。" $cGold | Out-Null
    AddCard $slide 332 336 258 142 (RGBColor 18 126 145) "系统看到的是问题" "拥堵是否加剧、设施是否破损、哪里存在风险、哪些区域需要更多公共资源。" $cMint | Out-Null
    AddCard $slide 612 336 296 142 (RGBColor 20 48 89) "城市智眼的意义" '让城市从“事后反应”转向“提前感知”，把分散的小事，连接成可治理的大系统。' $cCyan | Out-Null

    # Slide 3
    $slide = NewSlide $presentation 3
    AddRect $slide 1 0 0 960 540 $cBg 0 | Out-Null
    AddTextBox $slide 52 30 500 30 "2. 城市智眼靠什么运转？" 24 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 52 70 510 22 '不是一个摄像头更聪明，而是一套“测绘底座 + AI识别 + 空间推演”的组合。' 12 $cSoft "Microsoft YaHei UI" 0 | Out-Null

    AddRect $slide 5 52 120 368 284 (RGBColor 12 28 49) 0.02 $cBlue 0.8 | Out-Null
    AddTextBox $slide 76 142 320 24 "数字底座：先把城市测准" 18 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 76 186 320 158 "激光测绘、无人机航测、实景三维、BIM、GIS，共同构成一张持续更新的城市空间底图。\n\nBIM像建筑说明书，GIS像城市总地图，点云和三维模型让城市从平面走向立体。" 13 $cSoft "Microsoft YaHei UI" 0 | Out-Null
    AddRect $slide 5 76 356 96 26 $cBlue 0.04 | Out-Null
    AddRect $slide 5 186 356 96 26 $cCyan 0.04 | Out-Null
    AddRect $slide 5 296 356 96 26 $cMint 0.04 | Out-Null
    AddTextBox $slide 98 362 54 18 "激光点云" 12 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 208 362 54 18 "BIM" 12 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 316 362 54 18 "GIS" 12 $cWhite "Microsoft YaHei UI" -1 | Out-Null

    AddRect $slide 5 450 120 458 284 (RGBColor 13 40 67) 0.02 $cCyan 0.8 | Out-Null
    AddTextBox $slide 474 142 320 24 "算法引擎：再把城市看懂" 18 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 474 186 320 164 'AI视觉识别可以发现道路病害、违停、占道、井盖异常、人流车流变化。\n\n当这些结果叠加时间和空间信息后，系统就不只是在“看”，还会判断哪里风险更高、哪里需要提前干预。' 13 $cSoft "Microsoft YaHei UI" 0 | Out-Null
    AddRect $slide 5 474 356 140 26 $cGold 0.05 | Out-Null
    AddRect $slide 5 628 356 140 26 $cRed 0.05 | Out-Null
    AddTextBox $slide 496 362 118 18 "视觉识别" 12 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 650 362 118 18 "空间推演" 12 $cWhite "Microsoft YaHei UI" -1 | Out-Null

    # Slide 4
    $slide = NewSlide $presentation 4
    AddRect $slide 1 0 0 960 540 (RGBColor 244 248 252) 0 | Out-Null
    AddTextBox $slide 52 30 500 30 "3. 它在武汉能做什么？" 24 $cNavy "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 52 70 580 24 "把算法放进真实城市，就会形成从巡检到规划、从风险感知到便民服务的多场景能力。" 12 (RGBColor 78 96 120) "Microsoft YaHei UI" 0 | Out-Null

    AddCard $slide 52 120 400 150 (RGBColor 28 84 186) "道路智能巡检" '从“靠人找问题”变成“系统主动发现问题”，更快定位裂缝、坑洼、标线磨损等病害点位。' $cGold | Out-Null
    AddCard $slide 508 120 400 150 (RGBColor 15 124 150) "实景三维建模" "管理者可以在数字空间里直观看街区、看建筑、看道路，为改造、施工和规划提供可视化依据。" $cMint | Out-Null
    AddCard $slide 52 302 400 150 (RGBColor 18 49 92) "城市风险感知" "对内涝隐患、重点区域拥堵、施工风险、人流波动等进行更早预警，而不是等问题扩大后再应对。" $cCyan | Out-Null
    AddCard $slide 508 302 400 150 (RGBColor 34 97 115) "便民资源调配" "公交站点优化、停车引导、公共设施布局，都需要把空间数据和算法研判变成更贴近民生的服务。" $cGold | Out-Null

    # Slide 5
    $slide = NewSlide $presentation 5
    AddRect $slide 1 0 0 960 540 $cNavy 0 | Out-Null
    AddTextBox $slide 52 30 500 30 "4. 一座智慧城市，真正重要的是闭环" 24 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 52 72 550 22 '“城市智眼”不是单点技术，而是感知、理解、推演、响应四步接力。' 12 $cSoft "Microsoft YaHei UI" 0 | Out-Null

    $steps = @(
        @{X=72;  Y=170; Color=$cCyan; Title="感知"; Body="摄像头、测绘设备、无人机、激光雷达持续采集城市信息"},
        @{X=286; Y=170; Color=$cBlue; Title="识别"; Body="AI从海量图像和数据里找出车辆、行人、设施和异常事件"},
        @{X=500; Y=170; Color=$cMint; Title="研判"; Body="把结果放回GIS和实景三维场景中，结合历史数据做空间分析"},
        @{X=714; Y=170; Color=$cGold; Title="行动"; Body="把结论送到交通、城管、应急和规划部门，形成具体决策"}
    )

    foreach ($step in $steps) {
        AddRect $slide 9 $step.X $step.Y 132 132 $step.Color 0.02 | Out-Null
        AddTextBox $slide ($step.X + 34) ($step.Y + 26) 64 22 $step.Title 22 $cBg "Microsoft YaHei UI" -1 | Out-Null
        AddTextBox $slide ($step.X + 14) ($step.Y + 66) 104 48 $step.Body 10 $cBg "Microsoft YaHei UI" 0 | Out-Null
    }

    AddRect $slide 52 204 226 54 10 $cWhite 0.82 | Out-Null
    AddRect $slide 52 418 226 54 10 $cWhite 0.82 | Out-Null
    AddRect $slide 52 632 226 54 10 $cWhite 0.82 | Out-Null
    AddTextBox $slide 82 392 92 16 '从“看见”到“理解”' 11 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 296 392 92 16 '从“理解”到“预测”' 11 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 510 392 92 16 '从“预测”到“服务”' 11 $cWhite "Microsoft YaHei UI" -1 | Out-Null

    AddRect $slide 5 130 430 700 64 (RGBColor 10 26 46) 0.08 $cCyan 0.8 | Out-Null
    AddTextBox $slide 154 446 660 30 "真正改变城市效率的，不是某一个设备，而是这条完整的数据闭环。" 17 $cWhite "Microsoft YaHei UI" -1 | Out-Null

    # Slide 6
    $slide = NewSlide $presentation 6
    AddRect $slide 1 0 0 960 540 (RGBColor 249 250 252) 0 | Out-Null
    AddTextBox $slide 52 30 520 30 "5. 享受红利，也要看清挑战" 24 $cNavy "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 52 72 520 22 '智慧城市不是“越聪明越好”，而是要同时追求效率、安全与公平。' 12 (RGBColor 82 96 118) "Microsoft YaHei UI" 0 | Out-Null

    AddRect $slide 5 52 126 392 320 (RGBColor 233 247 242) 0 $cMint 0.8 | Out-Null
    AddTextBox $slide 74 146 120 22 "看得见的红利" 20 (RGBColor 20 108 83) "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 74 190 300 170 '更快发现道路和设施问题\n\n更早预警拥堵、积水、风险点\n\n更科学优化公交、停车和公共资源\n\n让城市治理从“被动处理”转向“主动服务”' 16 $cNavy "Microsoft YaHei UI" 0 | Out-Null

    AddRect $slide 5 516 126 392 320 (RGBColor 252 239 239) 0 $cRed 0.8 | Out-Null
    AddTextBox $slide 538 146 120 22 "必须正视的挑战" 20 (RGBColor 173 51 51) "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 538 190 312 170 "空间数据隐私：位置轨迹可能被滥用\n\n算法偏差：样本不完整会影响判断公平性\n\n数字壁垒：老人和弱势群体不能被技术门槛排除在外\n\n所以，技术要可监督、可解释、可约束" 16 $cNavy "Microsoft YaHei UI" 0 | Out-Null

    AddRect $slide 5 52 466 856 42 $cNavy 0 | Out-Null
    AddTextBox $slide 74 478 780 18 "结论：城市越智能，人越要更懂规则、更会保护数据，也更要参与监督算法如何被使用。" 13 $cWhite "Microsoft YaHei UI" 0 | Out-Null

    # Slide 7
    $slide = NewSlide $presentation 7
    AddRect $slide 1 0 0 960 540 $cBg 0 | Out-Null
    AddRect $slide 1 0 0 960 540 $cNavy 0.1 | Out-Null
    AddRect $slide 5 56 108 848 310 (RGBColor 15 34 57) 0.02 $cCyan 0.8 | Out-Null
    AddTextBox $slide 90 160 780 60 "算法正在重塑城市，\n但决定城市温度的，始终还是人。" 28 $cWhite "Microsoft YaHei UI" -1 | Out-Null
    AddTextBox $slide 90 280 720 64 '读懂武汉“城市智眼”的运行逻辑，既是理解智慧城市，也是学会在算法时代更科学、更安全地生活。' 16 $cSoft "Microsoft YaHei UI" 0 | Out-Null
    AddRect $slide 1 90 414 160 26 $cCyan 0.12 $cCyan 0.8 | Out-Null
    AddTextBox $slide 108 420 138 18 "谢谢聆听 / Q&A" 12 $cWhite "Microsoft YaHei UI" -1 | Out-Null

    $presentation.SaveAs($OutputPath)
    $presentation.Close()
    $pp.Quit()
}
finally {
    if ($presentation) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($pp) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
