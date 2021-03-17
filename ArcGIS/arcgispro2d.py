# # -*- coding: utf-8 -*-
# import arcpy
# import os
# # Sign in to portal
# arcpy.SignInToPortal('https://arcgis.ygwl.com/arcgis', 'gisgis', '1qazxsw2')
# # Set output file names
# outdir = r"D:\myarcgispro\protest"
# service = "Sharemapservice"
# sddraft_filename = service + ".sddraft"
# sddraft_output_filename = os.path.join(outdir, sddraft_filename)
# print (sddraft_output_filename)
# # 注册文件夹
# wrkspcs="D:\myarcgispro\protest"
# server_conn = "MY_HOSTED_SERVICES"
# if wrkspcs not in [i[2] for i in arcpy.ListDataStoreItems(server_conn, "FOLDER")]:
#      dsStatus = arcpy.AddDataStoreItem(server_conn, "FOLDER", "promapzc", wrkspcs, wrkspcs)
# print("注册: " + str(dsStatus))
# # Reference map to publish
# aprx = arcpy.mp.ArcGISProject(r"D:\myarcgispro\protest\protest.aprx")
# draft_value = aprx.listMaps("Map1")[0]
# # Create TileSharingDraft and set service properties
# sharing_draft = draft_value.getWebLayerSharingDraft("FEDERATED_SERVER", "MAP_IMAGE", service)
# sharing_draft.federatedServerUrl = "https://wl.arcgisonline.cn/server"
# sharing_draft.summary = "testpublish"
# sharing_draft.tags = "testpublish"
# sharing_draft.description = "credits"
# sharing_draft.credits = "testpublish"
# sharing_draft.useLimitations = "Limitation"
# print("description")
# # Create Service Definition Draft file
# sharing_draft.exportToSDDraft(sddraft_output_filename)
# print("to sd")
#
# # Stage Service
# try:
#     sd_filename = service + ".sd"
#     sd_output_filename = os.path.join(outdir, sd_filename)
#     print(sd_output_filename)
#     print(sddraft_output_filename)
#     arcpy.StageService_server(sddraft_output_filename, sd_output_filename)
#     warnings = arcpy.GetMessages(1)
#     print(warnings)
# except Exception as stage_exception:
#     print("Sddraft not staged. Analyzer errors - {}".format(str(stage_exception)))
# # Share to portal
# print("Uploading Service Definition")
# arcpy.UploadServiceDefinition_server(sd_output_filename, "https://wl.arcgisonline.cn/server")
# print("Successfully")


import arcpy
# from arcgis.gis import GIS

# 发布三维服务
portalUrl = "https://arcgis.ygwl.com/arcgis"
portalUsername = "gisgis"
portalpassword = "1qazxsw2"

arcpy.GetActivePortalURL()
print(arcpy.GetPortalInfo(portal_URL=arcpy.GetActivePortalURL()))

# 登录 portal
arcpy.SignInToPortal(portalUrl, portalUsername, portalpassword)
print("登录成功")

arcpy.MakeRasterLayer_management(r"E:\pycharm_project\tfservingconvert\gis\tiff\海南定安红外3result.tif", "study_quads_lyr")
# # arcpy.SaveToLayerFile_management("rdlayer", "tudyquadsLyr.lyr", "ABSOLUTE")
# # arcpy.MakeFeatureLayer_management(in_features, "study_quads_lyr", where_clause)
arcpy.SaveToLayerFile_management("study_quads_lyr", "studyquadsLyr.lyr", "ABSOLUTE")
# arcpy.PackageLayer_management(r"E:\pycharm_project\tfservingconvert\gis\studyquadsLyr.lyrx", "test.lpkx")

# arcpy.SharePackage_management(r"E:\pycharm_project\tfservingconvert\gis\test.lpkx", "gisgis", "1qazxsw2",
#                               "My Summary", "tag1, tag2", "My Credits",
#                               "MYGROUPS", "My Group")
# outputslpkpath = r"E:\pycharm_project\tfservingconvert\gis\test.lpkx"
# 切片包、矢量切片包和场景图层包。
# arcpy.SharePackage_management(outputslpkpath, portalUsername, portalpassword,
#                               "A01", "A01", "abb",
#                               "EVERYBODY"
#                               , None, "EVERYBODY", "TRUE",None)

# arcpy.SharePackage_management(outputslpkpath,portalUsername,portalpassword,"this is a summary","tag1, tag2","Credits","MYGROUPS",)

# arcpy.SharePackage_management(outputslpkpath, portalUsername, portalpassword,"测试定安05", "测试定安05","YGWL","EVERYBODY")
# gis = GIS(portalUrl, portalUsername, portalpassword, verify_cert=False)
# slpkitem = gis.content.add({}, data=outputslpkpath, folder='ygwl')
# slpkitem.publish()
print(arcpy.GetMessages())

print("成功上传slpk")



