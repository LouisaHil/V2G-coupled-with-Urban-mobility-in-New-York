This thesis explores the potential of vehicle-to-grid (V2G) technology in the context of New York City. Specifically, the work aims to model mobility behavior and offshore wind power to assess the feasibility of energy storage enabled by V2G. Firstly, I outline the methodology for simulating mobility patterns and the power output of wind farms. Subsequently, I outline a preliminary strategy to synchronize mobility, wind power availability, and the charging and discharging activities of electric vehicles (EVs). The thesis analyses several scenarios and identifies system limitations to ensure realistic simulations within computational constraints. These findings form the foundation for an innovative smart-charging algorithm, which examines the underlying dynamics governing EV fleets participating in V2G. Finally, I present a refined smart−charging algorithm calculating the actual number of EVs participating in V2G when integrating offshore wind power sources. The primary objective of this research is to gain a broad understanding of V2Gs true technical potential and comprehend the dynamics that govern this system. In conclusion, the study contributes to the sustainable development of coupled mobility and energy systems by providing insights into the potential of V2G technology to facilitate energy storage and reduce carbon emissions.











Introduction:
The world faces a severe climate crisis, which calls for a thorough reconsideration of our en- ergy systems and a determined pursuit of sustainable alternatives. We need to significantly transform our energy production and consumption habits to reduce global warming in the next three decades. This transformation involves decreasing our reliance on fossil fuels, increasing the use of low and zero-carbon energy sources, and exploring alternative energy options. One important aspect of shifting towards sustainable energy solutions is closely linked to our cities. Urban areas are crucial in the transition towards sustainable energy solutions because they are the epicenters of human activity. Cities offer many opportunities for innovative approaches to restructuring existing infrastructures. Therefore, creating sustainable cities for the future is essential in facilitating the necessary changes to combat the climate crisis.
All codes developed are sorted by chapter and can be accessed on Github: https://github.com/LouisaHil/V2G-coupled-with-Urban-mobility-in-New-York


1.1 Background Information and Introduction
Fortunately, there have been significant advancements in recent years thanks to the quick adop- tion of innovative technologies. The 2022 IPCC Report [2] on climate change provides a thorough technical analysis of the progress made in net-zero energy systems and highlights the current challenges in implementing them, emphasizing the need for further innovation and systematic changes [2]. Our research will focus on integrating renewable energy sources into the existing infrastructure of New York City.
The IPCC report [2] noted significant cost reductions from 2015 to 2020 in important energy systems mitigation options such as photovoltaics (PV), wind power, and batteries. Specifically, PV, wind energy, and batteries experienced declines of 56%, 45%, and 64% [2], respectively, increasing both the capacity and generation of PV and wind energy. Solar PV grew by 170%, reaching 680 TWh, while wind energy expanded by 70%, totaling 1420 TWh [2]. These devel- opments underscore the rapid adoption of renewable energy sources in recent years.
New York is ambitiously aiming for a clean energy transformation, targeting 70% of its electric- ity supply from renewable sources by 2030, doubling the current 31% [3]. By 2040, the goal is 100% [3]. To make this possible, New York is constructing an offshore wind farm with a capacity of 4,350 MW that will largely contribute to the 9,000 MW set goal by 2035 [4]. Currently, hy- droelectric power provides about 23% of the state’s electricity, whereas wind and solar will need to account for at least 45% to meet the 2030 goals [3]. If nuclear plants are not relicensed, wind and solar would have to supply about 67% of the electricity by 2040 [3]. Due to the unreliable
1
CHAPTER 1. INTRODUCTION 2: 
nature of wind and solar energy, the cost of this transformation includes the overbuilding of wind and solar energy facilities, substantial storage or backup energy production, and increased transmission costs [3]. Furthermore, the efficient use of renewable energy sources such as wind and solar power adds additional geographical and logistical challenges due to their remote loca- tions, necessitating better transmission [5]. Additionally, with the growing electricity demand given by widespread electrification, these transmission lines must increase their capacity to dis- tribute the necessary power. However, the power grid in the US has predominantly been based on a central station model, typically fueled by natural gas or hydropower, and has seen little significant change over the past century [5]. This requires grid reinforcement to facilitate the integration of distributed renewable energy sources. Achieving this while ensuring reliability is a complex task due to the grid’s current centralized design and transmission constraints. The Clean Energy Leadership Act of New York State sets a goal for 3,000 MW of battery storage by 2030 to help maintain grid stability [6]. However, the economic feasibility of this approach is hindered by high costs and the growing demand for raw materials needed for batteries. Battery storage costs must be reduced by approximately 90% to make them financially competitive with fossil fuels [6].
To achieve this ambitious plan, New York must enhance its grid design, build more infras- tructure, and introduce flexible resources to balance load with supply. This calls for carefully assessing energy storage solutions and robust transmission of high loads in the power grid.
Various energy storage systems exist, including flow batteries [7], compressed air energy stor- age [8], pumped hydroelectric storage [9], thermal energy storage [10], etc. that show potential solutions, but come with challenges. Each system has unique advantages and disadvantages in cost, scalability, lifespan, and efficiency. However, none of the latter energy storage systems are mobile and do not contribute significantly to the decentralization of the energy storage grid. Integrating mobility and decentralization in energy storage systems delivers key advantages, including enhanced grid resilience and improved efficiency by supporting dynamic demand re- sponse. Ultimately, shifting towards mobile, decentralized energy storage systems could lead to significant cost savings and a more robust, responsive, and equitable energy grid that facilitates the integration of renewable energy.
Vehicle to Grid (V2G) technology [11] is emerging and proposes a promising alternative to tackle these issues. V2G has existed for some time but has yet to be implemented on a large scale due to infrastructure constraints and a lack of economic incentives. V2G systems allow electric vehicles (EVs) to return energy to the grid during peak demand periods or when renewable energy production is low, providing a secondary utility beyond mobility. The potential of V2G technology has been the subject of much discussion, largely due to technical limitations, such as the possibility of reducing battery life. Although, some researchers suggest that under certain conditions, feeding energy back into the grid can potentially prolong the lifespan of batter- ies [12]. Additionally, implementing V2G technology would require a significant investment in bidirectional charging infrastructure. This includes high-speed DC charging stations with rates between 50-250 kW [13]. Currently, most charging stations operate at Level 1 and may not sup- port bidirectional power flow [14]. Significant EV participation in New York would necessitate a corresponding increase in charging stations, which must align with the local grid’s capacity. The grid’s capacity determines the power that can be drawn for EV charging and the amount that can be returned to the grid via V2G. Excessive demand could lead to power outages or infrastructure damage. Furthermore, other studies have concentrated on the socio-economic difficulties of integrating V2G, with private vehicles assuming a role beyond mobility. Encour- aging electric vehicle owners with economic incentives to charge and discharge their vehicles at specific times is a key factor in adopting vehicle-to-grid integration. Consequently, understand- ing and controlling vehicle mobility within the city’s dynamic nature becomes even more critical.

CHAPTER 1. INTRODUCTION 3: 
In New York City, the city’s energy and transportation systems are intertwined, and the in- creasing use of electric vehicles (EVs) highlights the need for a sophisticated approach to couple these. As EVs could account for 50-60% of transportation in the future [15], implementing vehicle-to-grid (V2G) technology can be transformative. Hence, our research centers around leveraging the mobility patterns of vehicles in New York City and coupling them with wind power generation to deduct the true technical potential of V2G.
The structure of this thesis begins with an outline of the methodology employed to simulate mo- bility patterns and the power generation of wind farms. Following this, a preliminary strategy is presented to interconnect mobility, the availability of wind power, and the charging and dis- charging activities of electric vehicles (EVs). These findings lay the groundwork for developing an innovative smart-charging algorithm, delving into the underlying dynamics that rule over EV fleets participating in V2G. In the process, we identify system limitations to ensure simulations are realistic and within computational constraints. In the concluding part of the research, a refined version of the smart-charging algorithm is presented, designed to calculate the actual number of EVs that participate in V2G when offshore wind power sources are integrated.
1.2 Literature Review on V2G and Renewable Energy Integration at the City Scale
Schla ̈pfer et al. (2021) [16] presented a model that leverages anonymized mobile location data to deduce population-wide mobility and incorporates a vehicle charging/discharging scheme. This approach permits a refined V2G energy supply/demand assessment, enabling better pho- tovoltaics deployment and infrastructure planning. The study emphasizes the importance of understanding human mobility patterns and urban transportation developments for effective V2G infrastructure planning.
Similarly, Xu et al. (2018) [17] utilized data from mobile phone activity, census records, and PEV charging behaviors to investigate the peak activity times and locations of PEV drivers. Their charging pattern analysis identified periods of high power demand that could potentially stress the power grid. To address this issue, they recommended a modified approach to PEV charging that accounts for the travel needs of individuals. The study’s results indicated that implementing these changes could reduce peak power demand by up to 47%.
Ghofrani et al. (2014) [18] suggested using a stochastic framework to manage the unpredictable nature of wind power. The researchers proposed using vehicle-to-grid (V2G) capabilities (EVs) to improve wind power prediction and reduce unpredictability. They achieved this by using an Auto Regressive Moving Average (ARMA) wind speed model and grouping EVs based on their daily driving habits. To minimize costs related to wind power imbalances and V2G expenses, the researchers utilized a Generic Algorithm (GA) in an optimization scheme. Despite the main focus of the study being economic cost optimization, the unique aspect of this research was the pairing of wind power and mobility patterns.
To align EV charging and discharging with renewable wind power production, we have developed an innovative deterministic approach that requires the development of new methods. In the upcoming chapters, we will provide a detailed description of our new perspective on assessing the potential of V2G in New York. It is crucial to acknowledge that our analysis in this thesis is focused solely on the temporal aspect.
